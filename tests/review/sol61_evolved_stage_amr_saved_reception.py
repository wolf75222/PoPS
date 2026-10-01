"""Offline AMR Stage science and durable-image reception; never imports PoPS.

ROOT supplies two external seals. No synthetic example enters receive().
The only reused module is an earlier independent stdlib/NumPy wire codec.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "stage_amr_independent_wire", Path(__file__).with_name("sol61_integral_feedback_offline_oracle.py"))
wire = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = wire
_spec.loader.exec_module(wire)
need, digest, strict_json = wire.require, wire.digest, wire.strict_json
DT, TOL = .01, 3e-8
CONTROLS = dict(tolerance=1e-10, max_iterations=20, linear_tolerance=1e-8,
                linear_max_iterations=240, restart=60, armijo=1e-4, minimum_step=1/1024)
QUALIFICATION = "homogeneous-original-composite-Q-durable@1"
PHASES = ("initial", "accepted", "continuous", "reloaded", "replay")
STEPS = dict(initial=0, accepted=1, continuous=2, reloaded=1, replay=2)
CASES = {(n, m) for n in (8, 16) for m in (1, 2)}


def exact(row, keys, label):
    need(type(row) is dict and set(row) == set(keys), label+" exact members differ")


def typed(a,b):
    return json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False) == json.dumps(b,sort_keys=True,separators=(",",":"),allow_nan=False)


def same(a, b, label):
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label+" bytes differ")


def finite_array(value, shape, label):
    need(type(value) is np.ndarray and value.dtype == np.dtype("float64")
         and value.shape == shape and np.isfinite(value).all(), label+" finite binary64 geometry differs")


def q_of(t):
    """Original physical accumulation, independently spelled component by component."""
    if t.shape[0] == 1:
        return np.stack((t[0]*(1+t[0]),))
    need(t.shape[0] == 2, "foreign witness width")
    a, b = t
    return np.stack((a+a*a+b*b/10, b+b*b+a*b/5))


def declared_load(width):
    need(type(width) is int and width in (1, 2), "foreign declared witness width")
    initial = np.array([.15, .25][:width], dtype=float)
    target = initial.copy()
    target[0] = .16
    if width == 2:
        # Original total-Q conservation determines a positive second temperature.
        total = sum(q_of(initial[:, None])[:, 0])
        rhs, linear = total-target[0]-target[0]**2, 1+target[0]/5
        target[1] = 2*rhs/(linear+math.sqrt(linear*linear+4*1.1*rhs))
    initial_q = q_of(initial[:, None])[:, 0]
    load = (q_of(target[:, None])[:, 0]-initial_q)/DT
    if width == 2:
        load[1] = -load[0]
    return initial, target, initial_q, load


def topology(boxes, n, ranks, modes, owners):
    """Derive finest masks from authenticated global boxes, not saved 'active'."""
    need(type(n) is int and n in (8, 16) and type(ranks) is int and ranks in (1, 2), "foreign topology witness")
    need(type(boxes) is np.ndarray and boxes.dtype == np.dtype("int64") and boxes.ndim == 2
         and boxes.shape[1] == 5, "AMR flattened boxes differ")
    present = [np.zeros((n*2**level,)*2, dtype=bool) for level in (0, 1)]
    counts = [0, 0]
    for level, x0, y0, x1, y1 in boxes.tolist():
        need(level in (0, 1) and 0 <= x0 <= x1 < n*2**level
             and 0 <= y0 <= y1 < n*2**level, "box level/bounds differ")
        view = present[level][y0:y1+1, x0:x1+1]
        need(not view.any(), "duplicate or overlapping global box ownership")
        view[:] = True
        counts[level] += 1
        if level == 1:
            need(x0 % 2 == y0 % 2 == 0 and x1 % 2 == y1 % 2 == 1, "fine box is not parent aligned")
    need(present[0].all() and present[1].any(), "missing base coverage or refined level")
    for level in (0, 1):
        need(modes[level] in ("replicated", "partitioned"), "unknown distribution mode")
        value = owners[level]
        need(value.dtype == np.dtype("int64") and value.ndim == 1, "owner-map type differs")
        need((value.size == 0 if modes[level] == "replicated" else value.size == counts[level])
             and all(0 <= int(rank) < ranks for rank in value), "owner-map authority/box count differs")
    covered = present[1].reshape(n, 2, n, 2).any(axis=(1, 3))
    masks = (present[0] & ~covered, present[1])
    need(masks[0].any(), "witness does not exercise partial composite coverage")
    volume = sum(Fraction(int(mask.sum()), (n*2**level)**2) for level, mask in enumerate(masks))
    need(volume == 1, "finest composite volume differs")
    return masks


def science(images, masks, n, width):
    """Original F = Q(T)-Q(previous)-dt*load; homogeneous div(D grad T)=0.

    Signed D=((.012,.002),(-.001,.014)) is not declared SPD. This witness
    cannot distinguish D from its transpose because every actual gradient is zero.
    """
    need(set(images) == set(PHASES), "missing/foreign scientific phase")
    _, target, initial_q, load = declared_load(width)
    names = [f"T{i}" for i in range(width)]+(["z"] if width == 2 else [])
    metrics = {}
    for phase in PHASES:
        step, amounts, residual, projection, constraint = STEPS[phase], [Fraction(0) for _ in range(width)], 0., 0., 0.
        need(len(images[phase]) == 2, "missing AMR level")
        for level, row in enumerate(images[phase]):
            size, mask = n*2**level, masks[level]
            expected_keys = {*(f"Q{i}" for i in range(width)), "forcing", "active"}
            if step:
                expected_keys |= set(names) | {name+"-previous" for name in names} | {"history_sample_identity_"+name for name in names}
            need(set(row) == expected_keys, "scientific level inventory differs")
            need(row["active"].dtype == np.dtype("bool") and row["active"].shape == mask.shape
                 and np.array_equal(row["active"], mask), "saved active mask differs from authenticated topology")
            q = np.stack([row[f"Q{i}"].reshape(size, size) for i in range(width)])
            f = row["forcing"].reshape(width+1, size, size)
            need(all(row[f"Q{i}"].shape == (size*size,) for i in range(width))
                 and row["forcing"].shape == ((width+1)*size*size,),"native flat component layout differs")
            finite_array(q, (width, size, size), "Q")
            finite_array(f, (width+1, size, size), "forcing")
            need(np.max(np.abs(f[:width, mask]-load[:, None])) <= TOL, "original prescribed load differs")
            if not step:
                need(np.max(np.abs(q[:, mask]-initial_q[:, None])) <= TOL, "declared initial Q differs")
                continue
            same(row["forcing"], images["initial"][level]["forcing"], "readonly forcing capture")
            t = np.stack([row[name].reshape(size, size) for name in names[:width]])
            need(all(row[name].shape == row[name+"-previous"].shape == (size*size,) for name in names),"native scalar history shape differs")
            finite_array(t, (width, size, size), "T")
            need((t[:, mask] > 0).all(), "foreign negative-temperature quadratic branch")
            need(np.max(np.abs(t[:, mask]-t[:, mask][:, :1])) <= TOL, "homogeneous temperature witness differs")
            previous = images["initial" if step == 1 else "accepted"][level]
            old = np.stack([previous[f"Q{i}"].reshape(size, size) for i in range(width)])
            residual = max(residual, float(np.max(np.abs((q-old-DT*f[:width])[:, mask]))))
            projection = max(projection, float(np.max(np.abs((q-q_of(t))[:, mask]))))
            if step == 1:
                need(np.max(np.abs(t[:, mask]-target[:, None])) <= TOL, "first original target differs")
            for name in names:
                sample = row["history_sample_identity_"+name]
                need(sample.dtype == np.dtype("uint8") and sample.ndim == 1,"saved history sample dtype differs")
                history_point(sample.tobytes(),name,level,step)
            if width == 2:
                z = row["z"].reshape(size, size)
                finite_array(z, (size, size), "z")
                constraint = max(constraint, float(np.max(np.abs((z-t[0]/4-t[1]/2)[mask]))))
            for i in range(width):
                amounts[i] += sum((Fraction(float(v)) for v in q[i, mask]), Fraction())/(size*size)
        if step:
            balance = [float(amount)-initial_q[i]-step*DT*load[i] for i, amount in enumerate(amounts)]
            need(max(residual, projection, constraint, *map(abs, balance)) <= TOL, "original composite Q equation/projection/balance failed")
            if width == 2:
                need(abs(float(sum(amounts))-sum(initial_q)) <= TOL, "original heat exchange is not conservative")
            metrics[phase] = dict(original_Q_linf=residual, Q_projection_linf=projection,
                                  z_constraint_linf=constraint, amounts=list(map(float, amounts)), balance=balance)
    for left, right in (("accepted", "reloaded"), ("continuous", "replay")):
        for a, b in zip(images[left], images[right], strict=True):
            for name in a:
                same(a[name], b[name], left+"/"+right+" "+name)
    for level in (0, 1):
        for name in names:
            same(images["continuous"][level][name+"-previous"], images["accepted"][level][name], "previous accepted temperature")
    return metrics


def diagnostic_image(raw, rank, ranks):
    need(raw[:8] == b"POPSDIA1" and len(raw) >= 40, "diagnostic codec header differs")
    width, owner, size, count = struct.unpack_from("<QQQQ", raw, 8)
    need((width, owner, size) == (64, rank, ranks) and count <= (len(raw)-40)//16, "diagnostic rank/width/count differs")
    at, entries, previous = 40, {}, None
    for _ in range(count):
        need(at+8 <= len(raw), "truncated diagnostic length")
        length = struct.unpack_from("<Q", raw, at)[0]
        at += 8
        need(length <= len(raw)-at-8, "diagnostic name exceeds image")
        name = raw[at:at+length]
        at += length
        bits = raw[at:at+8]
        at += 8
        need((previous is None or previous < name) and not name.startswith(b"pops.balance-term"), "diagnostic duplicate/order/reserved name")
        entries[name] = bits
        previous = name
    need(at == len(raw), "trailing diagnostic bytes")
    return entries


def rank_images(state, offsets, ranks, label):
    need(state.dtype == np.dtype("uint8") and state.ndim == 1 and offsets.dtype == np.dtype("int64")
         and offsets.shape == (ranks+1,) and offsets[0] == 0 and offsets[-1] == state.size
         and np.all(offsets[1:] >= offsets[:-1]), label+" rank offsets/type differ")
    return [state[int(offsets[i]):int(offsets[i+1])].tobytes() for i in range(ranks)]


def empty_exchange(raw):
    # Genuine AmrSystem::checkpoint_interval_exchanges exports legacy @1 when
    # no integral declarations exist. @2 is distinct, not a renamed @1 image.
    need(raw in (b"POPSEX01"+struct.pack("<Q",0), b"POPSEX02"+struct.pack("<QQQ",0,0,0)),
         "unexpected/nonempty accepted exchange ledger for this Stage witness")


def history_point(raw, name, level, step):
    encoded = name.encode()
    head = b"POPSHID1"+struct.pack("<Q", len(encoded))+encoded+struct.pack("<qQ", level, 2)
    need(raw[:len(head)] == head and len(raw) == len(head)+64, "AMR history ring/level/depth differs")
    for slot, start in enumerate((0., (step-1)*DT)):
        values = struct.unpack_from("<QQQQ", raw, len(head)+slot*32)
        expected = (2, int.from_bytes(struct.pack("<d", start), "little"),
                    int.from_bytes(struct.pack("<d", DT), "little"), 1)
        need(values == expected, "history publication point/duration/ordinal differs")


def envelope(arrays, artifact, bind, semantic):
    manifest = strict_json(str(arrays["pops_checkpoint_manifest"].item()))
    exact(manifest, ("schema_version", "runtime_kind", "semantic_identity", "artifact_identity", "bind_identity",
                     "run_identity", "clock", "arrays", "restart_identity"), "checkpoint envelope")
    need(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
         and manifest["runtime_kind"] == "amr", "run AMR envelope version differs")
    for name, expected in (("artifact", artifact), ("bind", bind), ("semantic", semantic)):
        need(wire.identity_token(manifest[name+"_identity"], name) == expected, "checkpoint "+name+" differs")
    need(set(arrays) == set(manifest["arrays"]) | {"pops_checkpoint_manifest", "pops_restart_identity"}, "checkpoint member inventory differs")
    for name, evidence in manifest["arrays"].items():
        a = arrays[name]
        need(not a.dtype.hasobject, "checkpoint object array")
        actual = dict(dtype=a.dtype.str, shape=list(a.shape), content_sha256=digest(wire.cbor(
            dict(protocol="pops.array-evidence.v1", dtype=a.dtype.str, shape=list(a.shape)))+a.tobytes(order="C")))
        need(evidence == actual, "checkpoint typed-array digest differs: "+name)
    base = {k:v for k,v in manifest.items() if k != "restart_identity"}
    token = wire.identity_token(manifest["restart_identity"], "restart")
    need(manifest["restart_identity"]["hexdigest"] == digest(wire.cbor(
        dict(protocol="pops.identity", domain="restart", schema_version=1, payload=base)))
        and str(arrays["pops_restart_identity"].item()) == token, "checkpoint restart seal differs")
    return manifest


def accepted_contract(contract,step):
    need(type(contract["schema_version"]) is int and contract["schema_version"] == 7
         and contract["guarantee"] == "bit_identical_accepted_state" and contract["program_state"] == "compiled",
         "native AMR accepted contract differs")
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


def checkpoint(arrays, phase, images, n, width, ranks, identities, abi):
    step = STEPS[phase]
    manifest = envelope(arrays, *identities)
    clock = dict(time=(step*DT).hex(), macro_step=step)
    need(manifest["clock"] == clock and wire.scalar(arrays["t"], "real").hex() == clock["time"]
         and wire.scalar(arrays["macro_step"], "int") == step, "checkpoint accepted clock differs")
    need(wire.scalar(arrays["pops_amr_checkpoint_version"], "int") == 11
         and str(arrays["abi_key"].item()) == abi and int(arrays["n_ranks"]) == ranks
         and int(arrays["n_levels"]) == int(arrays["configured_n_levels"]) == 2, "AMR durable envelope differs")
    geometry = strict_json(str(arrays["pops_spatial_contract"].item()))
    exact(geometry,("schema_version","dimension","shape","lower","upper","periodicity",
                    "refinement_ratios","native_layout_identity","identity"),"spatial contract")
    need(type(geometry["schema_version"]) is int and geometry["schema_version"] == 1
         and type(geometry["dimension"]) is int and geometry["dimension"] == 2
         and geometry["shape"] == [n,n] and all(type(v) is int for v in geometry["shape"])
         and geometry["lower"] == [0..hex()]*2 and geometry["upper"] == [1..hex()]*2
         and geometry["periodicity"] == [True,True] and all(type(v) is bool for v in geometry["periodicity"])
         and geometry["refinement_ratios"] == [[2,2]], "original unit-square AMR spatial geometry differs")
    payload = {k:v for k,v in geometry.items() if k != "identity"}
    need(geometry["identity"] == "pops.checkpoint-spatial-layout.v1:sha256:"+digest(wire.cbor(
        dict(protocol="pops.identity",domain="checkpoint-spatial-layout",schema_version=1,payload=payload))),
         "spatial contract identity differs")
    names = [f"T{i}" for i in range(width)]+(["z"] if width == 2 else [])
    need(list(arrays["blocks"]) == [*(f"Q{i}" for i in range(width)), "forcing"], "checkpoint block order differs")
    need(list(arrays["history_names"]) == sorted(names), "checkpoint history registry differs")
    masks = topology(arrays["patch_boxes"], n, ranks,
                     [str(arrays[f"distribution_mode_{l}"].item()) for l in (0, 1)],
                     [arrays[f"dmap_{l}"] for l in (0, 1)])
    for level in (0, 1):
        for name in [*(f"Q{i}" for i in range(width)), "forcing"]:
            same(arrays[f"state_{name}_{level}"], images[phase][level][name], "checkpoint physical state "+name)
        for name in names:
            need(int(arrays["history_depth_"+name]) == 2 and int(arrays["history_ncomp_"+name]) == 1
                 and list(arrays["history_levels_"+name]) == [0, 1]
                 and list(arrays["history_stored_slots_"+name]) == [0, 1], "history depth/width/levels/storage differs")
            need(arrays[f"history_init_{name}_level_{level}"].item() is True
                 and int(arrays[f"history_fill_count_{name}_level_{level}"]) == step, "history fill/initialization differs")
            same(arrays[f"history_slot_dt_{name}_level_{level}"], np.array([DT, DT]), "history durations")
            history_point(arrays[f"history_sample_identity_{name}_level_{level}"].tobytes(), name, level, step)
            same(arrays[f"history_sample_identity_{name}_level_{level}"],
                 images[phase][level]["history_sample_identity_"+name],"saved/durable history publication identity")
            for slot, key in ((0, name+"-previous"), (1, name)):
                same(arrays[f"history_{name}_level_{level}_{slot}"], images[phase][level][key], "checkpoint physical history slot")
        need(arrays[f"auxiliary_checkpoint_{level}"].dtype == np.dtype("uint8")
             and arrays[f"auxiliary_checkpoint_{level}"].size > 0, "native accepted auxiliary image missing")
    temporal = strict_json(str(arrays["temporal_restart_state"].item()))
    need(temporal["clock"] == clock and temporal["status"] == "accepted" and temporal["synchronized"] is True
         and temporal["strategy"] == dict(kind="fixed_dt", dt=dict(kind="binary64", value=DT.hex()))
         and temporal["controller_state"]["last_accepted_dt"] == DT.hex(), "temporal strategy/boundary differs")
    for key in ("clock_cursors","schedule_cursors","synchronization_cursors","history_cursors","cache_cursors"):
        need(type(temporal[key]) is dict,"temporal cursor table absent")
        for cursor in temporal[key].values():
            need(type(cursor) is dict and cursor.get("phase") == "accepted","temporal cursor is provisional")
            if "time" in cursor:
                need(cursor["time"] == clock["time"],"temporal cursor has stale time")
            if "macro_step" in cursor:
                need(cursor["macro_step"] == step,"temporal cursor has stale step")
    diagnostics = rank_images(arrays["program_diagnostics_state"], arrays["program_diagnostics_offsets"], ranks, "diagnostic")
    parsed = [diagnostic_image(raw, rank, ranks) for rank, raw in enumerate(diagnostics)]
    for row in parsed:
        residual = [struct.unpack("<d", bits)[0] for name, bits in row.items() if name.endswith(b".rel_residual")]
        need(len(residual) == 1 and math.isfinite(residual[0]) and 0 <= residual[0] <= CONTROLS["tolerance"], "native original-F diagnostic differs")
    for raw in rank_images(arrays["program_exchange_state"], arrays["program_exchange_offsets"], ranks, "exchange"):
        empty_exchange(raw)
    contract = strict_json(str(arrays["amr_accepted_contract"].item()))
    accepted_contract(contract,step)
    need(arrays["program_accepted_state"].dtype == np.dtype("uint8") and arrays["program_accepted_state"].size
         and arrays["program_accepted_state_source_authority"].dtype == np.dtype("uint8")
         and arrays["program_accepted_state_source_authority"].size, "opaque native Program authority missing")
    return manifest, masks, parsed


def token_data(token,domain):
    match = re.fullmatch(r"pops\."+domain+r"\.v([1-9][0-9]*):sha256:([0-9a-f]{64})",token)
    need(match is not None,"foreign identity token domain")
    return dict(domain=domain,schema_version=int(match[1]),algorithm="sha256",hexdigest=match[2])


def run_identity(envelope):
    exact(envelope,("protocol","kind","schema_version","payload"),"run envelope")
    need(envelope["protocol"] == "pops.manifest" and envelope["kind"] == "run"
         and type(envelope["schema_version"]) is int and envelope["schema_version"] == 3,"run schema differs")
    row = envelope["payload"]
    exact(row,("bind_identity","continuation_identity","start_time","start_macro_step","controls","run_identity"),"run payload")
    exact(row["controls"],("t_end","step_transaction","max_steps","output_mode"),"run controls")
    for value in (row["start_time"],row["controls"]["t_end"]):
        need(type(value) is float and math.isfinite(value),"run time is not finite binary64")
    need(type(row["start_macro_step"]) is int and type(row["controls"]["max_steps"]) is int,"run step type differs")
    value = dict(schema_version=3,bind_identity=token_data(row["bind_identity"],"bind"),
                 continuation_identity=None if row["continuation_identity"] is None else token_data(row["continuation_identity"],"run"),
                 start_time=row["start_time"].hex(),start_macro_step=row["start_macro_step"],
                 controls={**row["controls"],"t_end":row["controls"]["t_end"].hex()})
    expected = "pops.run.v1:sha256:"+digest(wire.cbor(dict(protocol="pops.identity",domain="run",schema_version=1,payload=value)))
    need(row["run_identity"] == expected,"run request digest differs")
    return row


def declared_source(amr_raw, equations_raw, controls_raw):
    """AST admission only; never executes/imports the author's oracle or helper."""
    trees = [ast.parse(raw.decode()) for raw in (amr_raw, equations_raw, controls_raw)]
    build = next(node for node in trees[0].body if isinstance(node, ast.FunctionDef) and node.name == "build")
    need(digest(ast.dump(build, include_attributes=False).encode()) ==
         "48ceca9d578ac14ca43571a8e8568ea3218d8685ff227ed3d405fe73a445f411",
         "declared original stage/captures/layout build AST differs; explicit source review required")
    dumps = {ast.dump(node, include_attributes=False) for node in ast.walk(trees[0])}
    fragments = (
        'q = [Reaction(temperature[0], 1+ValueExpr(temperature[0]))]',
        'q = [Reaction(a, 1+ValueExpr(a))+Reaction(b, .1*ValueExpr(b)), Reaction(b, 1+ValueExpr(b))+Reaction(a, .2*ValueExpr(b))]',
        'constraints = ({auxiliary: Reaction(auxiliary, 1)+Reaction(temperature[0], -.25)+Reaction(temperature[1], -.5) == 0} if auxiliary is not None else {})',
        'program.step_strategy(FixedDt(DT))',
        'program.store_history(u.name, observed[field[u]], depth=1, owner_block=blocks[0])',
    )
    for source in fragments:
        need(ast.dump(ast.parse(source).body[0], include_attributes=False) in dumps, "original source body/storage contract differs")
    expected = {"DT": ".01", "DENSE_BYTES": "256*1024**2", "ACCEPTANCE": "3e-8",
                "DIFFUSION": "np.array([[.012,.002],[-.001,.014]])", "FD_STEP": "1e-6",
                "CONTROLS": "dict(tolerance=1e-10,max_iterations=20,linear_tolerance=1e-8,linear_max_iterations=240,restart=60,armijo=1e-4,minimum_step=1/1024)"}
    assignments = {node.targets[0].id:node.value for tree in trees for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    for key, source in expected.items():
        need(key in assignments and ast.dump(assignments[key]) == ast.dump(ast.parse(source, mode="eval").body), "original source constant differs: "+key)


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
    expected_controls = {key:value if type(value) is int else dict(kind="binary64",value=value.hex()) for key,value in CONTROLS.items()}
    need(typed(attrs["newton_controls"],expected_controls) and attrs["finite_difference_step"] == dict(kind="binary64",value=(1e-6).hex())
         and type(attrs["ncomp"]) is int and attrs["ncomp"] == width+(width == 2)
         and attrs["right_preconditioner"] == "pops.amr.full-residual-basis-lu@1"
         and typed(attrs["right_preconditioner_resources"],dict(version=1,max_dense_bytes=dict(uint64_hex="0000000010000000"),
             scope="per_rank_dense_arrays_active_map_and_numeric_towers"))
         and attrs["source_contract"].get("temporal_tau") is not None and attrs["source_contract"].get("evolved_stage") is not None,
         "actual IR original seven controls/duration/product/realization differ")
    return actual


def contract():
    return dict(schema="sol61.evolved-stage-amr.owner-pins@1", qualification=QUALIFICATION,
                native_evidence=True, mode="serial or mpi2", ranks="1 or 2", source_commit="exact git40",
                native_build_source_commit="exact git40", abi_key="exact native ABI", ir_version="16 or 17 from actual source",
                roots=["absolute RO evidence/archive roots"], native="path+sha256", sdk="path+sha256",
                package_manifest="path+sha256", source_files={key:"path+sha256" for key in ("amr", "equations", "controls", "fixture")},
                junit="one path+sha256+rank per real rank; full raw batch XML", batch_names="exact full batch list",
                cases="exact 4 cells8/16 x widths1/2: receipt pin +closed files[path]=sha256+artifact/bind/semantic",
                approval=dict(schema="sol61.evolved-stage-amr.root-approval@1", approved_by="ROOT",
                              qualification=QUALIFICATION, pins_sha256="external pins SHA256"))


def receive(pins_path, pins_sha, approval_path, approval_sha):
    """No assembler/seal producer. ROOT supplies actual identities and exact inventory."""
    pins_raw, approval_raw = Path(pins_path).read_bytes(), Path(approval_path).read_bytes()
    need(digest(pins_raw) == pins_sha and digest(approval_raw) == approval_sha, "external ROOT seals differ")
    pins, approval = strict_json(pins_raw), strict_json(approval_raw)
    need(approval == dict(schema="sol61.evolved-stage-amr.root-approval@1", approved_by="ROOT",
                         qualification=QUALIFICATION, pins_sha256=pins_sha), "ROOT approval scope differs")
    exact(pins, ("schema", "qualification", "native_evidence", "mode", "ranks", "source_commit", "native_build_source_commit",
                 "abi_key", "ir_version", "roots", "native", "sdk", "package_manifest", "source_files", "junit", "batch_names", "cases"), "owner pins")
    need(pins["schema"] == "sol61.evolved-stage-amr.owner-pins@1" and pins["qualification"] == QUALIFICATION
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
        row_file(receipt["native"])
        need(receipt["native"]["sha256"] == pins["native"]["sha256"], "receipt native DSO differs")
        images = {phase:[wire.archive(row_file(row)) for row in receipt["phases"][phase]["levels"]] for phase in PHASES}
        arrays = {phase:wire.archive(row_file(receipt["checkpoints"][phase])) for phase in ("accepted", "continuous", "replay")}
        need(len({receipt["checkpoints"][p]["path"] for p in arrays}) == 3, "checkpoint overwrite aliases")
        program_hashes = []
        for component in receipt["compilation"]:
            for name in ("DSO", "sidecar"):
                row_file(component[name])
            if component["component"].startswith("program-"):
                program_hashes.append(program_image(strict_json(row_file(component["ir.json"])),
                    row_file(component["cpp"]).decode(), pins["ir_version"], component["program_hash"],width))
        need(len(program_hashes) == 1, "this same-layout witness requires one actual Program IR")
        manifests, masks, diagnostics = {}, None, {}
        for phase in arrays:
            need(str(arrays[phase]["program_hash"].item()) == program_hashes[0], "checkpoint installed Program differs")
            manifests[phase], actual_masks, diagnostics[phase] = checkpoint(arrays[phase], phase, images, n,width,pins["ranks"],
                (case["artifact"],case["bind"],case["semantic"]),pins["abi_key"])
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
