"""Offline finite-product reception. No PoPS import; no native execution/approval.

Only ROOT-approved actual file inventories can enter receive(). Protocol/math
unit tests use labelled synthetic bytes and never supply native evidence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction

import numpy as np


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = load("m19_saved_checkpoint_protocol", "sol61_integral_feedback_offline_oracle.py")
oracle = load("m19_saved_fraction_reference", "sol61_m19_product_oracle.py")
need, strict_json, digest = protocol.require, protocol.strict_json, protocol.digest
PHASES = ("initial", "accepted", "restored", "replayed")
NAMES = ("population", "integral", "weighted", "extended")
CASES = {(nx, nv, width, reverse) for nx, nv, width in ((4, 3, 3), (2, 5, 1), (7, 3, 5))
         for reverse in (False, True)}
TOL = 2e-12


def exact(row, keys, where):
    need(type(row) is dict and set(row) == set(keys), where + " exact keys differ")


def case_id(nx, nv, width, reverse):
    need(type(reverse) is bool and all(type(v) is int for v in (nx, nv, width)), "case types differ")
    need((nx, nv, width, reverse) in CASES, "foreign finite-product witness case")
    return f"{reverse}-{nx}-{nv}-{width}"


def canonical(path):
    path = Path(path).absolute()
    need(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
         "path escapes or aliases declared origin")
    need(path == path.resolve(strict=True), "path is not canonical")
    return path


def leaf(path):
    path = canonical(path)
    need(path.is_file() and path.stat().st_size <= protocol.MAX_FILE_BYTES, "missing/oversized actual file")
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def pinned(row, root):
    exact(row, ("path", "sha256"), "file pin")
    need(type(row["path"]) is str and type(row["sha256"]) is str
         and re.fullmatch("[0-9a-f]{64}", row["sha256"]) is not None, "file pin types differ")
    path = Path(row["path"])
    path = canonical(path if path.is_absolute() else Path(root) / path)
    need(path.is_relative_to(canonical(root)), "file escapes declared root")
    actual = leaf(path)
    need(actual["sha256"] == row["sha256"], "external file digest differs: " + str(path))
    return path, path.read_bytes()


def typed_array(array):
    value = np.ascontiguousarray(array) if array.shape else array
    header = protocol.cbor(dict(protocol="pops.array-evidence.v1", dtype=value.dtype.str, shape=list(value.shape)))
    return dict(dtype=value.dtype.str, shape=list(value.shape), content_sha256=digest(header + value.tobytes()))


def envelope(raw, phase, abi, artifact=None, bind=None):
    arrays = protocol.archive(raw)
    need({"t", "macro_step", "abi_key", "pops_checkpoint_manifest", "pops_restart_identity"} <= set(arrays),
         "checkpoint envelope absent")
    manifest = strict_json(str(arrays["pops_checkpoint_manifest"].item()))
    initial = phase == "initial"
    keys = {"schema_version", "runtime_kind", "semantic_identity", "artifact_identity", "bind_identity",
            "run_identity", "clock", "arrays", "restart_identity"} | ({"origin"} if initial else set())
    exact(manifest, keys, "checkpoint envelope")
    need(type(manifest["schema_version"]) is int and manifest["schema_version"] == (2 if initial else 1),
         "initial/run envelope version differs")
    need(type(manifest["runtime_kind"]) is str and bool(manifest["runtime_kind"]), "runtime kind absent")
    clock = {"time": protocol.scalar(arrays["t"], "real").hex(),
             "macro_step": protocol.scalar(arrays["macro_step"], "int")}
    need(manifest["clock"] == clock and str(arrays["abi_key"].item()) == abi, "checkpoint ABI/clock differs")
    for domain in ("semantic", "artifact", "bind"):
        token = protocol.identity_token(manifest[domain + "_identity"], domain)
        if domain == "artifact" and artifact is not None:
            need(token == artifact, "checkpoint artifact differs")
        if domain == "bind" and bind is not None:
            need(token == bind, "checkpoint bind differs")
    if initial:
        need(manifest["run_identity"] is None and manifest["origin"] == {"schema_version": 1, "kind": "bound_initial"}
             and clock == {"time": 0..hex(), "macro_step": 0}, "invented initial run provenance")
    else:
        protocol.identity_token(manifest["run_identity"], "run")
    need(type(manifest["arrays"]) is dict
         and set(arrays) == set(manifest["arrays"]) | {"pops_checkpoint_manifest", "pops_restart_identity"},
         "checkpoint member inventory differs")
    for name, evidence in manifest["arrays"].items():
        need(evidence == typed_array(arrays[name]), "checkpoint typed array digest differs")
    payload = {k: v for k, v in manifest.items() if k != "restart_identity"}
    restart = protocol.identity_token(manifest["restart_identity"], "restart")
    expected = digest(protocol.cbor(dict(protocol="pops.identity", domain="restart", schema_version=1, payload=payload)))
    need(manifest["restart_identity"]["hexdigest"] == expected
         and str(arrays["pops_restart_identity"].item()) == restart, "checkpoint restart digest differs")
    return arrays, manifest


def ownership(rows, ranks, nx, nv):
    need(type(rows) is list and len(rows) == ranks, "ownership rank inventory differs")
    modes = {}
    for name in NAMES:
        shape = (nx, nv) if name in ("population", "extended") else (1, nx)
        counts = []
        for row in rows:
            exact(row, NAMES, "rank ownership")
            need(type(row[name]) is list, "box inventory is not a list")
            count = np.zeros(shape, dtype=np.int64)
            for box in row[name]:
                need(type(box) is list and len(box) == 2
                     and all(type(point) is list and len(point) == 2 for point in box)
                     and all(type(v) is int for point in box for v in point), "box type differs")
                (a0, b0), (a1, b1) = box
                # Native axis zero is v in the product, x in the retained layout.
                need(0 <= a0 < a1 <= shape[1] and 0 <= b0 < b1 <= shape[0], "box escapes physical support")
                count[b0:b1, a0:a1] += 1
            need(np.all(count <= 1), "rank-local duplicate owner")
            counts.append(count)
        stacked = np.stack(counts)
        need(np.all(stacked.sum(axis=0) == 1) or np.all(stacked == 1), "missing or duplicated distributed owner")
        modes[name] = "distributed" if np.all(stacked.sum(axis=0) == 1) else "replicated"
    return modes


def original(initial, phases, nx, nv, width):
    shapes = {"population": (width, nx, nv), "integral": (width, 1, nx),
              "weighted": (width, 1, nx), "extended": (width, nx, nv)}
    for state in (initial, *phases):
        exact(state, NAMES, "physical state")
        for name, value in state.items():
            need(value.dtype == np.dtype("float64") and value.shape == shapes[name]
                 and np.isfinite(value).all(), "state shape/component/finitude differs")
    population = initial["population"]
    uniform = oracle.reduce_exact(population, ("x", "v"), ("x",), {"v": (Fraction(4, nv),) * nv})[:, None, :]
    signed = oracle.reduce_exact(population, ("x", "v"), ("x",),
                                 {"v": tuple(Fraction((-1) ** j * (j + 1)) for j in range(nv))})[:, None, :]
    lifted = oracle.lift(signed[:, 0, :], ("x",), ("x", "v"), (nx, nv))
    expected = dict(zip(NAMES, (population, uniform, signed, lifted), strict=True))
    errors = {}
    for index, state in enumerate(phases):
        for name in NAMES:
            difference = np.abs(state[name] - expected[name])
            need(np.isfinite(difference).all() and np.all(difference <= TOL), "original finite equation differs: " + name)
            errors[f"{index}:{name}"] = float(difference.max())
    need(all(phases[0][name].tobytes() == phases[1][name].tobytes() for name in NAMES),
         "restored state is not bit-identical to accepted")
    return errors


def closed_phases(directory):
    directory = canonical(directory)
    files, expected = {}, {directory / "provenance.json"}
    files["provenance"] = leaf(directory / "provenance.json")
    for phase in PHASES:
        state, receipt = directory / (phase + "-state.npz"), directory / (phase + "-receipt.json")
        row = strict_json(receipt.read_bytes())
        name = row.get("checkpoint")
        need(type(name) is str and name and Path(name).name == name and "\\" not in name,
             "checkpoint basename escapes case")
        checkpoint = directory / name
        need(checkpoint not in expected and checkpoint not in (state, receipt), "checkpoint alias/reuse")
        files[phase] = dict(state=leaf(state), receipt=leaf(receipt), checkpoint=leaf(checkpoint))
        need(row["saved_state_sha256"] == files[phase]["state"]["sha256"]
             and row["checkpoint_sha256"] == files[phase]["checkpoint"]["sha256"], "phase receipt digest differs")
        expected.update((state, receipt, checkpoint))
    need(set(directory.iterdir()) == expected and len(expected) == 13, "case closed 13-file inventory differs")
    return files


def junit(raw, rank, ranks, cases):
    need(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "JUnit external declarations forbidden")
    root = ET.fromstring(raw)
    tests = list(root.iter("testcase"))
    need(len(tests) == 6 and not any(list(t.iter("failure")) + list(t.iter("error")) + list(t.iter("skipped")) for t in tests),
         "JUnit requires six passing cases without failure/error/skip")
    for suite in root.iter("testsuite"):
        need(suite.get("tests") == str(len(list(suite.iter("testcase"))))
             and all(suite.get(k, "0") == "0" for k in ("failures", "errors", "skipped")), "JUnit suite counters differ")
    seen = set()
    for test in tests:
        need(test.get("classname") == "tests.python.integration.runtime.test_m19_product_support_runtime",
             "JUnit source realm differs")
        name = test.get("name", "")
        need(name.startswith("test_native_product_reduce_lift_restart[") and name.endswith("]"), "foreign JUnit case")
        key = name[len("test_native_product_reduce_lift_restart["):-1]
        need(key in cases and key not in seen, "missing/duplicate JUnit case")
        seen.add(key)
        rows = list(test.iter("property"))
        values = {row.get("name"): row.get("value") for row in rows}
        need(len(values) == len(rows) and None not in values, "JUnit duplicate/malformed properties")
        need(values.get("native_dimension") == "2" and values.get("mpi_rank") == str(rank)
             and values.get("mpi_size") == str(ranks)
             and values.get("native_sha256") == cases[key]["native_sha256"]
             and values.get("saved_receipts") == cases[key]["directory"], "JUnit rank/native/realm differs")
    need(seen == set(cases), "JUnit case set differs")


def checkpoint(raw, receipt, state, phase, abi):
    arrays, manifest = envelope(raw, phase, abi, receipt["artifact_identity"], receipt["bind_identity"])
    need(manifest["runtime_kind"] == "multi_layout_uniform", "checkpoint is not a product-layout container")
    need(set(arrays) == {"t", "macro_step", "abi_key", "layout_ids", "mapping_evaluations", "runtime_consumer_graph",
                         "runtime_consumer_cursors", "runtime_consumer_diagnostics", "pops_checkpoint_manifest",
                         "pops_restart_identity", "layout_checkpoint_0", "layout_checkpoint_1", "layout_checkpoint_2"},
         "composite checkpoint container inventory differs")
    ids = list(map(str, arrays["layout_ids"]))
    need(len(ids) == 3 and len(set(ids)) == 3, "three real layouts required")
    need(strict_json(str(arrays["mapping_evaluations"].item())) == receipt["mapping_counts"], "checkpoint map report differs")
    found = {}
    for index in range(3):
        image = arrays["layout_checkpoint_%d" % index]
        need(image.dtype == np.dtype("uint8") and image.ndim == 1, "child checkpoint storage differs")
        child, child_manifest = envelope(image.tobytes(), phase, abi)
        need(child_manifest["runtime_kind"] == "uniform" and child_manifest["clock"] == manifest["clock"],
             "child layout/clock differs")
        for block in map(str, child["blocks"]):
            need(block in NAMES and block not in found, "child block missing/foreign/duplicate")
            width = state[block].shape[0]
            need(protocol.scalar(child["ncomp_" + block], "int") == width
                 and list(child["names_" + block]) == ["quantity_%d" % c for c in range(width)], "child component names differ")
            actual = child["state_" + block]
            need(actual.dtype == state[block].dtype and actual.shape == state[block].shape
                 and actual.tobytes() == state[block].tobytes(), "checkpoint saved physical state differs")
            found[block] = ids[index]
        if "temporal_restart_state" in child:
            temporal = strict_json(str(child["temporal_restart_state"].item()))
            need(temporal["clock"] == manifest["clock"], "child temporal clock differs")
    need(set(found) == set(NAMES) and found["integral"] == found["weighted"]
         and len({found["population"], found["integral"], found["extended"]}) == 3, "physical layouts collapsed")
    return manifest


def metadata_decode(value):
    if type(value) is dict:
        if "bytes_hex" in value:
            exact(value, ("bytes_hex",), "metadata bytes tag")
            raw = value["bytes_hex"]
            need(type(raw) is str and re.fullmatch("(?:[0-9a-f]{2})*", raw) is not None,
                 "noncanonical metadata bytes tag")
            return bytes.fromhex(raw)
        return {key: metadata_decode(item) for key, item in value.items()}
    if type(value) is list:
        return [metadata_decode(item) for item in value]
    need(value is None or type(value) in (str, bool, int, float), "unknown metadata type")
    if type(value) is float:
        need(np.isfinite(value), "nonfinite metadata")
    return value


def identity_cbor(value):
    def head(major, number):
        need(0 <= number < 2 ** 64, "identity CBOR length overflow")
        if number < 24:
            return bytes([major * 32 + number])
        for width, marker in ((1, 24), (2, 25), (4, 26), (8, 27)):
            if number < 2 ** (width * 8):
                return bytes([major * 32 + marker]) + number.to_bytes(width, "big")
        raise ValueError("identity CBOR length overflow")
    if type(value) is bytes:
        return head(2, len(value)) + value
    if type(value) is list:
        return head(4, len(value)) + b"".join(identity_cbor(item) for item in value)
    if type(value) is dict:
        need(all(type(k) is str for k in value), "identity CBOR key type differs")
        rows = sorted(((protocol.cbor(k), identity_cbor(v)) for k, v in value.items()), key=lambda row: (len(row[0]), row[0]))
        return head(5, len(rows)) + b"".join(k + v for k, v in rows)
    return protocol.cbor(value)


def identity_hash(domain, payload):
    return digest(identity_cbor(dict(protocol="pops.identity", domain=domain, schema_version=1, payload=payload)))


def artifact_payload(provenance):
    plan = provenance["compiled_plan"]
    # The retained value is CompiledPlanRecord._payload, not the original
    # ResolvedSimulationPlan._payload. Preserve that distinction.
    reference = plan["resolved_plan_identity"]
    exact(reference, ("domain", "schema_version", "algorithm", "digest"), "resolved plan reference")
    need(reference["domain"] == "resolved-plan" and reference["algorithm"] == "sha256"
         and type(reference["schema_version"]) is int and reference["schema_version"] == 1
         and type(reference["digest"]) is bytes and len(reference["digest"]) == 32, "resolved plan reference differs")
    payload = dict(schema_version=2, plan_identity=reference,
                   target=plan["target"], platform_manifest=provenance["platform"], components=provenance["compiled_components"])
    expected = "pops.artifact.v1:sha256:" + identity_hash("artifact", payload)
    need(provenance["artifact_identity"] == expected, "actual artifact aggregate payload digest differs")
    need(type(plan["resolved_dimension"]) is int and plan["resolved_dimension"] == 2, "plan dimension differs")
    return reference["digest"].hex()


def physical_maps(payload, nv):
    phase = dict(kind="physical_support", coordinates=[["velocity", "velocity-interval"], ["position", "periodic-position"]])
    physical = dict(kind="physical_support", coordinates=[["position", "periodic-position"]])
    unit = dict(kind="physical_dimension", powers=[])
    def reduction(weights):
        return dict(kind="support-reduction@2", source_support=phase, target_support=physical,
                    reductions=[dict(axis=0, domain=dict(lower=[-2, 1], upper=[2, 1], cells=nv),
                                     rule="explicit-cell-average-weighted-sum@1", weights=weights, dimension=unit)],
                    storage=dict(native_dimension=2, source_axes=[0, 1], target_axes=[0], source_to_target=[-1, 0],
                                 hidden_cells=1, hidden_measure=1, extension="constant", inverse_closure=False))
    w = Fraction(4, nv)
    expected = [reduction([[w.numerator, w.denominator]] * nv),
                reduction([[(-1) ** j * (j + 1), 1] for j in range(nv)]),
                dict(kind="support-extension@2", source_support=physical, target_support=phase, reductions=[],
                     storage=dict(native_dimension=2, source_axes=[0], target_axes=[0, 1], source_to_target=[1, -1],
                                  hidden_cells=1, hidden_measure=1, extension="constant", inverse_closure=False))]
    found = []
    def visit(row):
        if type(row) is dict:
            if row.get("kind") in ("support-reduction@2", "support-extension@2"):
                found.append(row)
            for item in row.values():
                visit(item)
        elif type(row) is list:
            for item in row:
                visit(item)
    visit(payload)
    def encode(row):
        return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    need({encode(row) for row in found} == {encode(row) for row in expected},
         "physical map axes/domains/units/quadratures differ")


def source_route(raw):
    text = raw.decode("utf-8")
    need('#include <pops/runtime/dynamic/physical_support_transfer.hpp>' in text
         and 'return pops::component::apply_physical_support_transfer(descriptor, request, status);' in text,
         "provider does not call actual common lowering")


def origin_records(data, cases):
    exact(data, ("schema", "source_commit", "native_build_source_commit", "roots", "python_package", "sdk", "native",
                 "sources", "common_lowering", "system_packages", "provider_sources", "generated_cpp", "cpp_dso_links"),
          "execution origins")
    need(data["schema"] == "sol61.m19-execution-owner@1", "execution origin schema differs")
    need(type(data["source_commit"]) is str and re.fullmatch("[0-9a-f]{40}", data["source_commit"]), "observed source commit absent")
    need(data["native_build_source_commit"] is None or type(data["native_build_source_commit"]) is str
         and re.fullmatch("[0-9a-f]{40}", data["native_build_source_commit"]), "native build source field differs")
    exact(data["roots"], ("installation", "runtime", "source"), "origin roots")
    roots = {name: canonical(path) for name, path in data["roots"].items()}
    for name in ("python_package", "sdk", "native"):
        pinned(data[name], roots["installation"])
    need(type(data["sources"]) is list and len(data["sources"]) == 2, "two actual fixture/helper sources required")
    source_paths = [str(pinned(row, roots["source"])[0]) for row in data["sources"]]
    need(len(set(source_paths)) == 2, "duplicate fixture/helper sources")
    for table in ("system_packages", "provider_sources", "generated_cpp", "cpp_dso_links"):
        exact(data[table], cases, table)
    pinned(data["common_lowering"], roots["installation"])
    for key in cases:
        for role in ("system_packages", "provider_sources"):
            rows = data[role][key]
            count = 4 if role == "system_packages" else 3
            need(type(rows) is list and len(rows) == count, "actual " + role + " inventory differs")
            paths = [str(pinned(row, roots["runtime"])[0]) for row in rows]
            need(len(set(paths)) == count, "duplicate " + role)
            if role == "provider_sources":
                for row in rows:
                    source_route(pinned(row, roots["runtime"])[1])
        rows = data["generated_cpp"][key]
        need(rows is None or type(rows) is list and len(rows) > 0, "generated CPP must be actual leaves or null")
        if rows is not None:
            paths = [pinned(row, roots["runtime"])[0] for row in rows]
            need(all(p.suffix == ".cpp" for p in paths) and len(set(paths)) == len(paths), "generated CPP alias/duplicate/type")
        links = data["cpp_dso_links"][key]
        need(links is None or type(links) is list and len(links) > 0, "CPP/DSO links must be records or null")
        if links is not None:
            need(rows is not None, "link proof lacks retained CPP")
            for link in links:
                exact(link, ("cpp", "dso", "build_receipt"), "CPP/DSO link")
                need(link["cpp"] in rows and link["dso"] in data["system_packages"][key], "link references foreign source/binary")
                pinned(link["build_receipt"], roots["runtime"])
    return roots


def assemble(base, mode, directories, junits, origins):
    base = canonical(base)
    ranks = 1 if mode == "serial" else 2 if mode == "mpi2" else 0
    need(ranks and len(directories) == 6 and len(junits) == ranks, "exact serial/MPI2 six-case inventory required")
    cases = {}
    for directory in directories:
        directory = canonical(directory)
        need(directory.is_relative_to(base), "case directory escapes archive")
        files = closed_phases(directory)
        provenance = strict_json(Path(files["provenance"]["path"]).read_bytes())
        dims = provenance["dimensions"]
        exact(dims, ("nx", "nv", "components"), "dimensions")
        # Reverse is an actual pytest parameter, never inferred from a plan hash.
        names = set()
        for path in junits:
            root = ET.fromstring(canonical(path).read_bytes())
            for test in root.iter("testcase"):
                values = {row.get("name"): row.get("value") for row in test.iter("property")}
                if values.get("saved_receipts") == str(directory):
                    names.add(test.get("name", ""))
        candidates = [case_id(dims["nx"], dims["nv"], dims["components"], reverse) for reverse in (False, True)]
        keys = [key for key in candidates if "test_native_product_reduce_lift_restart[" + key + "]" in names]
        need(len(keys) == 1 and keys[0] not in cases, "case parameter/directory binding differs")
        cases[keys[0]] = dict(dimensions=dims, reverse=keys[0].startswith("True-"), directory=str(directory), files=files,
                             native_sha256=provenance["native_sha256"])
    result = dict(schema="sol61.m19-owner-pins@1", mode=mode, ranks=ranks, archive_root=str(base),
                  execution_origins=origins, junit=[leaf(path) for path in junits], cases=cases)
    validate_inventory(result)
    return result


def validate_inventory(pins):
    exact(pins, ("schema", "mode", "ranks", "archive_root", "execution_origins", "junit", "cases"), "pins")
    need(pins["schema"] == "sol61.m19-owner-pins@1" and type(pins["ranks"]) is int
         and (pins["mode"], pins["ranks"]) in (("serial", 1), ("mpi2", 2)), "mode/rank types differ")
    base = canonical(pins["archive_root"])
    need(type(pins["cases"]) is dict and len(pins["cases"]) == 6, "six cases required")
    seen_dirs = set()
    for key, case in pins["cases"].items():
        exact(case, ("dimensions", "reverse", "directory", "files", "native_sha256"), "case")
        dims = case["dimensions"]
        exact(dims, ("nx", "nv", "components"), "dimensions")
        need(key == case_id(dims["nx"], dims["nv"], dims["components"], case["reverse"]), "case key differs")
        directory = canonical(case["directory"])
        need(directory.is_relative_to(base) and directory not in seen_dirs, "duplicate/foreign case directory")
        seen_dirs.add(directory)
        need(case["files"] == closed_phases(directory), "closed phase pins differ")
    origin_records(pins["execution_origins"], pins["cases"])
    need(type(pins["junit"]) is list and len(pins["junit"]) == pins["ranks"], "JUnit rank inventory differs")
    paths = []
    for rank, row in enumerate(pins["junit"]):
        path, raw = pinned(row, base)
        paths.append(path)
        junit(raw, rank, pins["ranks"], pins["cases"])
    need(len(set(paths)) == len(paths), "JUnit rank file reused")


def receive(pins_path, pins_sha, approval_path, approval_sha):
    raw, approved = canonical(pins_path).read_bytes(), canonical(approval_path).read_bytes()
    need(digest(raw) == pins_sha and digest(approved) == approval_sha, "external seal digest differs")
    approval = strict_json(approved)
    exact(approval, ("schema", "approved_by", "pins_sha256"), "ROOT approval")
    need(approval == dict(schema="sol61.m19-root-approval@1", approved_by="ROOT", pins_sha256=pins_sha),
         "ROOT has not approved this exact manifest")
    pins = strict_json(raw)
    validate_inventory(pins)
    reports = {}
    origins = pins["execution_origins"]
    source_sha = {row["path"]: row["sha256"] for row in origins["sources"]}
    for key, case in pins["cases"].items():
        base, files = Path(case["directory"]), case["files"]
        provenance = strict_json(pinned(files["provenance"], base)[1])
        exact(provenance, ("schema", "native_path", "native_sha256", "abi_key", "python_package", "native_capabilities",
                           "artifact_identity", "platform", "compiled_components", "compiled_plan", "dimensions",
                           "physical_equations", "source_sha256", "metadata_encoding"), "native provenance")
        need(provenance["metadata_encoding"] == "json-with-bytes-hex.v1", "metadata encoding differs")
        decoded = metadata_decode(provenance)
        plan_digest = artifact_payload(decoded)
        physical_maps(decoded["compiled_plan"], case["dimensions"]["nv"])
        blocks = decoded["compiled_components"]["blocks"]
        need(len(blocks) == 4 and {row["name"] for row in blocks} == set(NAMES), "component block inventory differs")
        need({row["binary"]["binary_sha256"] for row in blocks}
             == {row["sha256"] for row in origins["system_packages"][key]}, "actual System binary hashes differ")
        need(provenance["schema"] == "sol61.m19-product-native@1"
             and provenance["native_path"] == origins["native"]["path"]
             and provenance["native_sha256"] == origins["native"]["sha256"] == case["native_sha256"]
             and provenance["python_package"] == origins["python_package"]["path"]
             and provenance["source_sha256"] == source_sha
             and provenance["dimensions"] == case["dimensions"]
             and provenance["physical_equations"] == "finite reduction and constant extension only",
             "native/package/source/physical provenance differs")
        states, receipts, layouts = {}, {}, {}
        dims = case["dimensions"]
        nx, nv, width = dims["nx"], dims["nv"], dims["components"]
        for phase in PHASES:
            row = files[phase]
            state = protocol.archive(pinned(row["state"], base)[1])
            receipt = strict_json(pinned(row["receipt"], base)[1])
            exact(receipt, ("schema", "phase", "time", "time_hex", "macro_step", "artifact_identity", "bind_identity",
                            "mapping_counts", "local_boxes_by_rank", "checkpoint", "checkpoint_sha256", "saved_state_sha256"), "phase receipt")
            time, step = {"initial": (0., 0), "accepted": (.01, 1), "restored": (.01, 1), "replayed": (.02, 2)}[phase]
            need(receipt["schema"] == "sol61.m19-product-state@1" and receipt["phase"] == phase
                 and type(receipt["time"]) is float and receipt["time"].hex() == time.hex() == receipt["time_hex"]
                 and type(receipt["macro_step"]) is int and receipt["macro_step"] == step
                 and receipt["artifact_identity"] == provenance["artifact_identity"], "phase/identity/clock differs")
            counts = receipt["mapping_counts"]
            need(type(counts) is dict and len(counts) == 3 and all(type(k) is str and k and type(v) is int and v == step
                                                                for k, v in counts.items()), "mapping cadence differs")
            layouts[phase] = ownership(receipt["local_boxes_by_rank"], pins["ranks"], nx, nv)
            checkpoint(pinned(row["checkpoint"], base)[1], receipt, state, phase, provenance["abi_key"])
            states[phase], receipts[phase] = state, receipt
        need(all(set(row["mapping_counts"]) == set(receipts["initial"]["mapping_counts"]) for row in receipts.values()),
             "mapping identity changed across phases")
        need(receipts["restored"]["bind_identity"] == receipts["replayed"]["bind_identity"], "restart bind changed")
        errors = original(states["initial"], [states[p] for p in PHASES[1:]], nx, nv, width)
        reports[key] = dict(max_original_error=max(errors.values()), owner_modes=layouts,
                            aggregate_scope="recomputed_actual_artifact_payload", resolved_plan_sha256=plan_digest,
                            original_resolved_plan_payload_scope="not_stored", compiled_plan_record_scope="ROOT_owner_attested",
                            retained_cpp=origins["generated_cpp"][key] is not None,
                            cpp_dso_link_scope="owner_supplied_build_receipts" if origins["cpp_dso_links"][key] else "not_stored")
    return dict(scope="finite_product_saved_states_authsource_scope", cases=reports,
                source_commit=origins["source_commit"], native_build_source_commit=origins["native_build_source_commit"],
                native_build_source_scope="not_stored" if origins["native_build_source_commit"] is None else "ROOT_owner_attested",
                strong_cpp_dso_link_qualified=False, limitation="No Vlasov/BGK/field solve or new native execution; build receipts are ROOT-attested, not independently proven compile graphs.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pending = sub.add_parser("assemble")
    pending.add_argument("--archive-root", required=True)
    pending.add_argument("--mode", choices=("serial", "mpi2"), required=True)
    pending.add_argument("--case-directory", action="append", required=True)
    pending.add_argument("--junit", action="append", required=True)
    pending.add_argument("--execution-origins", required=True)
    pending.add_argument("--output", required=True)
    check = sub.add_parser("check")
    for name in ("pins", "pins-sha256", "approval", "approval-sha256"):
        check.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.command == "assemble":
        value = assemble(args.archive_root, args.mode, args.case_directory, args.junit,
                         strict_json(Path(args.execution_origins).read_bytes()))
        with Path(args.output).open("x") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        print("pending external ROOT approval; no native reception performed")
    else:
        print(json.dumps(receive(args.pins, args.pins_sha256, args.approval, args.approval_sha256), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
