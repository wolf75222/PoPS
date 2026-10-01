"""Saved captured-D reception: NumPy/Fraction/stdlib, no PoPS/native execution.

Assembly produces a pending inventory, never approval. Two external ROOT seals
are required. Synthetic protocol tests are not native evidence.
"""
from __future__ import annotations
import argparse
import ast
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np

spec = importlib.util.spec_from_file_location("captured_d_wire", Path(__file__).with_name("sol61_m19_saved_reception.py"))
wire = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wire
spec.loader.exec_module(wire)
protocol = wire.protocol
need, exact, digest = wire.need, wire.exact, wire.digest


def strict_json(raw):
    value = wire.strict_json(raw)
    def finite(node):
        if type(node) is float:
            need(math.isfinite(node), "nonfinite JSON number")
        elif type(node) in (dict, list):
            for item in node.values() if type(node) is dict else node:
                finite(item)
    finite(value)
    return value
PHASES = ("accepted", "continuous", "reloaded", "replay")
CLOCKS = dict(accepted=(.01, 1), continuous=(.02, 2), reloaded=(.01, 1), replay=(.02, 2))
CASES = {"scalar1": (1, [0]), "coupled3-201": (3, [2, 0, 1])}
DT, TOL = .01, 3e-8
CONTROLS = dict(tolerance=1e-10, max_iterations=20, linear_tolerance=1e-8,
                linear_max_iterations=240, restart=60, armijo=1e-4, minimum_step=1/1024)
QUALIFICATION = "saved-states-original-residual@1"
MAX_BINARY_BYTES = 1024*1024*1024  # offline DSO hashing budget, not a runtime limit


def matrices(width):
    """Independent exact original coefficients: no SPD or symmetrization."""
    need(type(width) is int and width in (1, 3), "foreign witness width")
    d, r = ([[".012"]], [["1.1"]]) if width == 1 else (
        [[".012", ".002", "0"], ["-.001", ".014", "0"], [".001", "0", "0"]],
        [["1.1", ".03", "-.01"], ["-.02", "1.3", ".02"], [".01", "-.03", "1.5"]])
    return tuple(tuple(Fraction(v) for v in row) for row in d), tuple(tuple(Fraction(v) for v in row) for row in r)


def prescribed(n, width):
    """Declared centre samples; no claim of continuum cell-average quadrature."""
    centres = (np.arange(n, dtype=np.float64)+.5)/n
    xx, yy = np.meshgrid(centres, centres, indexing="xy")
    q = np.stack([.15+.025*np.cos(2*np.pi*(i+1)*xx)+.02*np.sin(2*np.pi*yy) for i in range(width)])
    return q, .25*np.sin(2*np.pi*xx)+.15*np.cos(2*np.pi*yy)


def original(q, alpha, *, diffusion=None, harmonic=False):
    """Visit each periodic face once, add its equal/opposite FV contributions.

    Plane axis0 is physical y, axis1 x. Area/volume/gradient gives N**2 on
    the unit square. All signed, nonsymmetric, singular matrix entries survive.
    """
    width, ny, nx = q.shape
    need(alpha.shape == (ny, nx), "material geometry differs")
    d, r = matrices(width)
    d = np.array(d if diffusion is None else diffusion, dtype=float)
    spatial = np.zeros_like(q)
    for y in range(ny):
        for x in range(nx):
            for yy, xx, n in (((y+1) % ny, x, ny), (y, (x+1) % nx, nx)):
                a, b = 1+alpha[y, x], 1+alpha[yy, xx]
                mean = 2*a*b/(a+b) if harmonic else (a+b)/2
                flux = (d @ (q[:, yy, xx]-q[:, y, x]))*mean*n**2
                spatial[:, y, x] -= flux
                spatial[:, yy, xx] += flux
    return spatial+np.einsum("ij,jyx->iyx", np.array(r, dtype=float), q)+float(Fraction(1, 5))*q**3, spatial


def array(value, shape, label):
    need(type(value) is np.ndarray and value.dtype == np.dtype("float64") and value.shape == shape
         and np.isfinite(value).all(), "invalid finite binary64 array: "+label)


def same(a, b, label):
    need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label+" differs in bytes")


def science(initial, states, width, n=16):
    exact(initial, ("response", "forcing", "material", "target"), "initial NPZ")
    q0, alpha0 = prescribed(n, width)
    for key in ("response", "forcing", "target"):
        array(initial[key], (width, n, n), key)
    array(initial["material"], (1, n, n), "material")
    need(np.max(np.abs(initial["target"]-q0)) < 2e-14 and np.max(np.abs(initial["material"][0]-alpha0)) < 2e-14,
         "initial material/target differs from declared centre-sample recipe")
    need(not np.any(initial["response"].view(np.uint64)), "initial response is not canonical zero")
    forcing, _ = original(q0, initial["material"][0])
    need(np.max(np.abs(forcing-initial["forcing"])) < 2e-12, "initial forcing differs from original declared operator")
    exact(states, PHASES, "phase inventory")
    reports = {}
    for phase, saved in states.items():
        exact(saved, ("response", "forcing", "material", "solution", "time", "step"), "phase NPZ")
        time, step = CLOCKS[phase]
        need(protocol.scalar(saved["time"], "real").hex() == time.hex()
             and protocol.scalar(saved["step"], "int") == step, "phase exact clock differs")
        for key in ("response", "forcing", "solution"):
            array(saved[key], (width, n, n), key)
        array(saved["material"], (1, n, n), "material")
        for key in ("forcing", "material"):
            same(saved[key], initial[key], "readonly capture "+key)
        q = saved["solution"]
        need(all(np.ptp(component) > .07 for component in q), "constant/incorrectly permuted solution")
        lhs, spatial = original(q, saved["material"][0])
        residual = float(np.linalg.norm(lhs-saved["forcing"])/np.linalg.norm(saved["forcing"]))
        error = float(np.max(np.abs(q-q0)))
        consumption = float(np.max(np.abs(saved["response"]/(step*DT)-q)))
        need(error < TOL, "solution differs from independent declared target")
        need(residual < TOL, "original captured-D residual exceeds fixed tolerance")
        need(consumption < TOL, "true exterior consumer duration/value differs")
        need(float(np.max(np.abs(spatial))) > .01, "nonconstant diffusion action absent")
        reports[phase] = dict(solution_linf=error, original_residual_relative_l2=residual,
                              consumer_linf=consumption, spatial_action_linf=float(np.max(np.abs(spatial))))
    for a, b in (("accepted", "reloaded"), ("continuous", "replay")):
        for key in states[a]:
            same(states[a][key], states[b][key], a+"/"+b+" "+key)
    return reports


def read(path, *, budget=protocol.MAX_FILE_BYTES):
    path = wire.canonical(path)  # reject file/parent aliases and .. before resolve
    fd = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        need(before.st_size <= budget, "file exceeds reception budget")
        raw = os.pread(fd, before.st_size+1, 0)
        after = os.fstat(fd)
        def image(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
        need(image(before) == image(after) and len(raw) == before.st_size, "file drift during bounded read")
        return path, raw
    finally:
        os.close(fd)


def leaf(path):
    path, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def pinned(row, roots, *, budget=protocol.MAX_FILE_BYTES):
    exact(row, ("path", "sha256"), "file pin")
    need(type(row["path"]) is str and type(row["sha256"]) is str
         and re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None, "invalid file pin")
    path, raw = read(row["path"], budget=budget)
    need(any(path.is_relative_to(wire.canonical(root)) for root in roots), "file outside approved roots")
    need(digest(raw) == row["sha256"], "external file digest differs")
    return path, raw


def declared_source(raw):
    """Literal D/R verification in ROOT-pinned original declaration, no execution."""
    tree = ast.parse(raw)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "matrices"), None)
    need(fn is not None, "original matrix recipe absent")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "array"]
    def encode(rows):
        return tuple(tuple(Fraction(str(v)) for v in row) for row in rows)
    actual = [encode(ast.literal_eval(n.args[0])) for n in calls]
    expected = [encode(np.array(rows, dtype=float).tolist()) for width in (1, 3) for rows in matrices(width)]
    need(sorted(actual, key=repr) == sorted(expected, key=repr), "source original D/R canonicals differ")
    build = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"), None)
    need(build is not None, "physical declaration absent")
    names = {n.id for n in ast.walk(build) if isinstance(n, ast.Name)}
    literals = {n.value for n in ast.walk(build) if isinstance(n, ast.Constant) and type(n.value) is str}
    need({"DivCoeffGrad", "Reaction", "FieldProblem"} <= names and "Arithmetic@1" in literals,
         "original physical field/arithmetic declaration absent")
    # Versioned source-witness contract, not a model dispatch or execution.
    critical = (
        'lhs = Reaction(unknowns[row], .2*ValueExpr(unknowns[row])**2)',
        'lhs += Reaction(unknowns[column], float(reaction[row, column]))',
        'lhs -= DivCoeffGrad(unknowns[column], float(diffusion[row, column])*(1+material[0]))',
        'equations.append(lhs == load[row])',
        'source = fluid.source("actual_field_response", on=response, value=auxiliaries)',
        'captures = {blocks[1][load]: forcing.n}',
        'captures[blocks[2][material]] = coefficient.n',
        'request = field.bind_program_inputs(program=program, values=captures, at=program.stage("frozen-original-coefficients", c=0), solver=solver)',
        'outputs = tuple(observed[field[unknown]] for unknown in unknowns)',
        'rhs = program.rhs(state=current.n, fields=publication, terms=[SourceTerm(blocks[0][module.operator_handle("actual_field_response")])])',
        'program.commit(current.next, program.value("response-update", current.n+program.dt*rhs, at=current.next.point))',
    )
    nodes = {ast.dump(n, include_attributes=False) for n in ast.walk(build)}
    need(all(ast.dump(ast.parse(statement).body[0], include_attributes=False) in nodes for statement in critical),
         "source original body/capture point/consumer contract differs")


def history_point(raw, name, step):
    header = b"POPSHID1"+struct.pack("<Q", len(name))+name.encode()+struct.pack("<qQ", -1, 1)
    need(raw.startswith(header) and len(raw) == len(header)+32, "history identity header differs")
    kind, start, interval, ordinal = struct.unpack("<QQQQ", raw[len(header):])
    need(kind == 2 and start == int.from_bytes(struct.pack("<d", (step-1)*DT), "little")
         and interval == int.from_bytes(struct.pack("<d", DT), "little") and ordinal == 1,
         "stale captured/history publication point")


def checkpoint(raw, phase, saved, artifact, abi):
    arrays, manifest = wire.envelope(raw, "accepted", abi, artifact=artifact)
    need(manifest["runtime_kind"] == "uniform", "foreign checkpoint runtime")
    time, step = CLOCKS[phase]
    need(protocol.scalar(arrays["t"], "real").hex() == time.hex()
         and protocol.scalar(arrays["macro_step"], "int") == step, "checkpoint exact phase clock differs")
    geometry = strict_json(str(arrays["pops_spatial_contract"].item()))
    exact(geometry, ("schema_version", "dimension", "shape", "lower", "upper", "periodicity",
                     "refinement_ratios", "native_layout_identity", "identity"), "spatial contract")
    need(type(geometry["schema_version"]) is int and geometry["schema_version"] == 1
         and type(geometry["dimension"]) is int and geometry["dimension"] == 2
         and geometry["shape"] == [16, 16] and geometry["lower"] == [0..hex()]*2
         and geometry["upper"] == [1..hex()]*2 and geometry["periodicity"] == [True, True]
         and geometry["refinement_ratios"] == [] and all(type(v) is bool for v in geometry["periodicity"])
         and all(type(v) is int for v in geometry["shape"]), "checkpoint coordinate/axis/periodicity geometry differs")
    need(type(geometry["native_layout_identity"]) is str
         and re.fullmatch(r"pops\.native-spatial-layout\.v1:sha256:[0-9a-f]{64}", geometry["native_layout_identity"]) is not None,
         "native layout identity absent")
    need(geometry["identity"] == "pops.checkpoint-spatial-layout.v1:sha256:"+wire.identity_hash("checkpoint-spatial-layout", {k: v for k, v in geometry.items() if k != "identity"}),
         "spatial identity differs")
    width = saved["solution"].shape[0]
    need(list(arrays["blocks"]) == ["response", "forcing", "material"], "checkpoint block order differs")
    for block, names in (("response", [f"u{i}" for i in range(width)]),
                         ("forcing", [f"f{i}" for i in range(width)]), ("material", ["alpha"])):
        need(list(arrays["names_"+block]) == names and protocol.scalar(arrays["ncomp_"+block], "int") == len(names), "component ordering differs")
        actual = arrays["state_"+block]
        need(actual.dtype == np.dtype("float64") and actual.size == len(names)*16**2, "checkpoint physical array differs")
        same(actual.reshape(len(names), 16, 16), saved[block], "checkpoint state "+block)
    need(list(arrays["history_names"]) == [f"q{i}" for i in range(width)], "history registry differs")
    for i in range(width):
        name = f"q{i}"
        need(protocol.scalar(arrays["history_depth_"+name], "int") == 1 and protocol.scalar(arrays["history_ncomp_"+name], "int") == 1
             and arrays["history_init_"+name].item() is True and protocol.scalar(arrays["history_fill_count_"+name], "int") == 1
             and arrays["history_stored_slots_"+name].dtype == np.dtype("int64")
             and list(arrays["history_stored_slots_"+name]) == [0], "history initialization/storage differs")
        same(arrays["history_slot_dt_"+name], np.array([DT]), "history outgoing duration")
        sample = arrays["history_sample_identity_"+name]
        need(sample.dtype == np.dtype("uint8") and sample.ndim == 1, "history point storage differs")
        history_point(sample.tobytes(), name, step)
        value = arrays["history_"+name+"_0"]
        need(value.dtype == np.dtype("float64") and value.size == 16**2, "history field geometry differs")
        same(value.reshape(16, 16), saved["solution"][i], "observed history solution")
    need({"program_exchange_state", "program_exchange_offsets", "temporal_restart_state", "auxiliary_checkpoint"} <= set(arrays),
         "exact continuation images absent")
    temporal = strict_json(str(arrays["temporal_restart_state"].item()))
    need(temporal["clock"] == dict(time=time.hex(), macro_step=step) and temporal["status"] == "accepted"
         and temporal["synchronized"] is True and temporal["controller_state"]["last_accepted_dt"] == DT.hex(),
         "temporal accepted boundary/window differs")
    for key in ("clock_cursors", "schedule_cursors", "synchronization_cursors", "history_cursors", "cache_cursors"):
        need(type(temporal[key]) is dict, "temporal cursor map absent")
        for cursor in temporal[key].values():
            need(type(cursor) is dict and cursor.get("phase") == "accepted", "temporal cursor not accepted")
            if "time" in cursor:
                need(cursor["time"] == time.hex(), "temporal cursor point differs")
            if "macro_step" in cursor:
                need(cursor["macro_step"] == step, "temporal cursor step differs")
    return arrays


def junit(raw, rank, size, cases):
    root = ET.fromstring(raw)
    need(not root.findall(".//failure") and not root.findall(".//error") and not root.findall(".//skipped"), "JUnit failure/error/skip")
    for suite in root.iter("testsuite"):
        for name in ("failures", "errors", "skipped"):
            if name in suite.attrib:
                need(suite.attrib[name] == "0", "JUnit declared failure/error/skip count")
        if "tests" in suite.attrib:
            need(suite.attrib["tests"] == str(len(suite.findall("testcase"))), "JUnit declared test count differs")
    tests = root.findall(".//testcase")
    need(len(tests) == 2, "JUnit requires two actual cases per rank")
    seen = set()
    for test in tests:
        need(test.get("classname", "").endswith("test_public_captured_diffusion"), "foreign JUnit class")
        rows = test.findall("./properties/property")
        need(len({p.get("name") for p in rows}) == len(rows), "duplicate JUnit property")
        props = {p.get("name"): p.get("value") for p in rows}
        matches = [k for k, c in cases.items() if props.get("captured_diffusion_receipt") == c["receipt"]["path"]]
        need(len(matches) == 1 and matches[0] not in seen, "JUnit receipt absent/duplicate")
        key = matches[0]
        suffix = "1-order0" if key == "scalar1" else "3-order1"
        need(test.get("name") == "test_public_captured_diffusion_nonconstant_saved_and_exact_replay["+suffix+"]", "JUnit parameters differ")
        need(props.get("dimension") == "2" and props.get("rank") == str(rank)
             and props.get("size") == str(size) and props.get("artifact_identity") == cases[key]["artifact"]
             and props.get("evidence_path") == cases[key]["directory"], "JUnit actual provenance differs")
        seen.add(key)
    need(seen == set(CASES), "JUnit incomplete")


def case_inventory(directory):
    directory = wire.canonical(directory)
    receipt = strict_json(read(directory/"receipt.json")[1])
    files = {directory/"receipt.json"}
    initial = leaf(receipt["initial_npz"])
    files.add(Path(initial["path"]))
    need(initial["sha256"] == receipt["initial_sha256"], "initial receipt digest differs")
    exact(receipt["phases"], PHASES, "receipt phases")
    phases = {}
    for phase in PHASES:
        row = receipt["phases"][phase]
        exact(row, ("npz", "sha256", "checks"), "receipt phase")
        phases[phase] = leaf(row["npz"])
        need(phases[phase]["sha256"] == row["sha256"], "phase receipt digest differs")
        files.add(Path(phases[phase]["path"]))
    exact(receipt["checkpoints"], ("accepted", "continuous", "replay"), "checkpoint phases")
    cps = {k: leaf(row["path"]) for k, row in receipt["checkpoints"].items()}
    need(all(cps[k] == receipt["checkpoints"][k] for k in cps), "checkpoint receipt digest differs")
    files.update(Path(row["path"]) for row in cps.values())
    need(type(receipt["sources"]) is list, "compiler source inventory absent")
    for row in receipt["sources"]:
        exact(row, ("component", "path", "sha256"), "compiler source receipt")
    sources = [leaf(row["path"]) for row in receipt["sources"]]
    need(len(sources) == 1 and sources[0]["sha256"] == receipt["sources"][0]["sha256"], "retained Program CPP absent")
    files.update(Path(row["path"]) for row in sources)
    need(len(files) == 10 and all(p.parent == directory for p in files) and set(directory.iterdir()) == files,
         "closed captured-D case inventory differs")
    return dict(directory=str(directory), artifact=receipt["artifact"], receipt=leaf(directory/"receipt.json"),
                initial=initial, phases=phases, checkpoints=cps, sources=sources)


def origins(value, roots):
    exact(value, ("schema", "source_commit", "native_build_source_commit", "abi_key", "python_package", "sdk", "native", "sources", "cpp_dso_links"), "owner")
    need(value["schema"] == "sol61.captured-d-execution-owner@1", "owner schema differs")
    for key in ("source_commit", "native_build_source_commit"):
        need((value[key] is None and key == "native_build_source_commit") or
             (type(value[key]) is str and re.fullmatch("[0-9a-f]{40}", value[key]) is not None), "owner commit differs")
    need(type(value["abi_key"]) is str and value["abi_key"], "ABI absent")
    for key in ("python_package", "sdk", "native"):
        pinned(value[key], roots, budget=MAX_BINARY_BYTES if key == "native" else protocol.MAX_FILE_BYTES)
    exact(value["sources"], ("fixture", "physical_helper"), "source origins")
    declared_source(pinned(value["sources"]["physical_helper"], roots)[1])
    pinned(value["sources"]["fixture"], roots)
    need(value["cpp_dso_links"] is None, "unreviewed CPP-to-DSO link format cannot certify linking")


def assemble(root, directories, junits, owner, roots):
    root = wire.canonical(root)
    need(len(directories) == 2 and len(junits) in (1, 2), "two cases and Serial/MPI2 required")
    need(type(roots) is list and len(roots) == len(set(roots)) and str(root) in roots, "approved roots differ")
    origins(owner, roots)
    cases = {}
    for directory in directories:
        case = case_inventory(directory)
        need(Path(case["directory"]).is_relative_to(root), "case escapes archive")
        receipt = strict_json(pinned(case["receipt"], roots)[1])
        key = next((k for k, (w, o) in CASES.items() if (receipt["width"], receipt["order"]) == (w, o)), None)
        need(key is not None and key not in cases, "foreign/duplicate case")
        cases[key] = case
    need(set(cases) == set(CASES), "incomplete case coverage")
    junit_pins = [leaf(p) for p in junits]
    for rank, row in enumerate(junit_pins):
        junit(pinned(row, roots)[1], rank, len(junits), cases)
    return dict(schema="sol61.captured-d-owner-pins@1", qualification=QUALIFICATION, archive_root=str(root), file_roots=roots,
                mode="serial" if len(junits) == 1 else "mpi2", ranks=len(junits), owner=owner, junit=junit_pins, cases=cases)


def receive(pins_path, pins_sha, approval_path, approval_sha):
    raw, approved = read(pins_path)[1], read(approval_path)[1]
    need(digest(raw) == pins_sha and digest(approved) == approval_sha, "external seal differs")
    approval = strict_json(approved)
    exact(approval, ("schema", "approved_by", "pins_sha256", "qualification"), "approval")
    need(approval == dict(schema="sol61.captured-d-root-approval@1", approved_by="ROOT", pins_sha256=pins_sha, qualification=QUALIFICATION), "ROOT approval scope differs")
    pins = strict_json(raw)
    exact(pins, ("schema", "qualification", "archive_root", "file_roots", "mode", "ranks", "owner", "junit", "cases"), "pins")
    need(pins["schema"] == "sol61.captured-d-owner-pins@1" and pins["qualification"] == QUALIFICATION
         and type(pins["ranks"]) is int and pins["ranks"] in (1, 2)
         and pins["mode"] == ("serial" if pins["ranks"] == 1 else "mpi2"), "owner scope differs")
    roots = pins["file_roots"]
    need(type(roots) is list and pins["archive_root"] in roots and len(roots) == len(set(roots)), "file roots differ")
    origins(pins["owner"], roots)
    exact(pins["cases"], CASES, "sealed cases")
    need(len(pins["junit"]) == pins["ranks"] and len({r["path"] for r in pins["junit"]}) == pins["ranks"], "JUnit rank inventory differs")
    for rank, row in enumerate(pins["junit"]):
        junit(pinned(row, roots)[1], rank, pins["ranks"], pins["cases"])
    reports = {}
    for key, case in pins["cases"].items():
        need(Path(case["directory"]).is_relative_to(wire.canonical(pins["archive_root"])), "case outside archive root")
        need(case_inventory(case["directory"]) == case, "sealed case inventory differs")
        receipt = strict_json(pinned(case["receipt"], roots)[1])
        exact(receipt, ("kind", "artifact", "dimension", "rank", "size", "cells", "width", "order", "face_policy", "newton", "fd_step",
                        "solution_tolerance", "residual_tolerance", "native", "platform", "binaries", "sources", "initial_npz", "initial_sha256",
                        "phases", "checkpoints", "exact_restart_and_replay"), "receipt")
        width, order = CASES[key]
        need(receipt["kind"] == "actual-native-captured-D-original-MMS" and receipt["dimension"] == 2 and receipt["cells"] == 16
             and receipt["width"] == width and receipt["order"] == order and receipt["rank"] == 0 and receipt["size"] == pins["ranks"], "receipt case differs")
        need(all(type(receipt[k]) is int for k in ("dimension", "cells", "width", "rank", "size")), "receipt exact integer types differ")
        need(type(receipt["order"]) is list and all(type(i) is int for i in receipt["order"]), "permutation exact integer types differ")
        exact(receipt["newton"], CONTROLS, "Newton controls")
        need(all(type(receipt["newton"][k]) is type(v) for k, v in CONTROLS.items()), "Newton controls exact types differ")
        need(receipt["face_policy"] == "pops.field.face-mean.arithmetic@1" and receipt["newton"] == CONTROLS and receipt["fd_step"] == 1e-6
             and receipt["solution_tolerance"] == TOL and receipt["residual_tolerance"] == TOL and receipt["exact_restart_and_replay"] is True, "method/guards differ")
        need(receipt["native"] == pins["owner"]["native"], "selected native owner differs")
        need(len(receipt["binaries"]) == 4 and len({r["component"] for r in receipt["binaries"]}) == 4
             and {r["component"] for r in receipt["binaries"] if r["component"].startswith("block-")}
             == {"block-response", "block-forcing", "block-material"}, "compiled components differ")
        for row in receipt["binaries"]:
            exact(row, ("component", "path", "sha256", "compile_command"), "binary receipt")
            pinned({k: row[k] for k in ("path", "sha256")}, roots, budget=MAX_BINARY_BYTES)
        source = receipt["sources"][0]
        need(source["component"].startswith("program-") and source["component"] in {r["component"] for r in receipt["binaries"]},
             "compiler source component owner differs")
        cpp = pinned(case["sources"][0], roots)[1].decode("utf-8")
        signature = r"apply_general_field\s*<\s*pops::kNativeDimension\s*,\s*%d\s*,\s*%d\s*,\s*true\s*>" % (width, width**2)
        need(re.search(signature, cpp) is not None and "nonfinite_original_field_residual" in cpp
             and "field expression inputs require exact layout/distribution identity" in cpp,
             "retained compiler source original/capture/arithmetic route absent")
        initial = protocol.archive(pinned(case["initial"], roots)[1])
        states = {phase: protocol.archive(pinned(row, roots)[1]) for phase, row in case["phases"].items()}
        reports[key] = science(initial, states, width)
        images = {phase: checkpoint(pinned(row, roots)[1], phase, states[phase], receipt["artifact"], pins["owner"]["abi_key"])
                  for phase, row in case["checkpoints"].items()}
        checkpoint(pinned(case["checkpoints"]["accepted"], roots)[1], "reloaded", states["reloaded"], receipt["artifact"], pins["owner"]["abi_key"])
        skip = {"pops_checkpoint_manifest", "pops_restart_identity"}
        need(set(images["continuous"]) == set(images["replay"]), "replay checkpoint inventory differs")
        for name in set(images["continuous"])-skip:
            same(images["continuous"][name], images["replay"][name], "replay continuation "+name)
    return dict(qualification=QUALIFICATION, cases=reports, cpp_dso_link_qualified=False,
                gaps=["original artifact/Program aggregate payload absent", "block compiler CPP absent",
                      "independent restored checkpoint absent", "in-memory carrier/diagnostic comparison images absent",
                      "private PreparedFieldCapture owner/lease/point image not persisted; history point is checked separately",
                      "no convergence/AMR/arbitrary-D solvability/SPD/empty-MPI partition qualification"]
                     + (["native build source commit unavailable"] if pins["owner"]["native_build_source_commit"] is None else []),
                evidence="externally ROOT-approved files; checker performs no native execution")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("assemble", help="pending inventory only")
    for name in ("archive-root", "owner", "output"):
        a.add_argument("--"+name, required=True)
    for name in ("case", "junit", "file-root"):
        a.add_argument("--"+name, action="append", required=True)
    r = sub.add_parser("receive")
    for name in ("pins", "pins-sha256", "approval", "approval-sha256"):
        r.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    if args.command == "assemble":
        data = assemble(args.archive_root, args.case, args.junit, strict_json(read(args.owner)[1]), args.file_root)
        encoded = json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+"\n"
        target = Path(args.output)
        need(not target.exists() or target.read_text() == encoded, "existing pending inventory differs")
        target.write_text(encoded)
        print("pending external ROOT approval; no native qualification")
    else:
        print(json.dumps(receive(args.pins, args.pins_sha256, args.approval, args.approval_sha256), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
