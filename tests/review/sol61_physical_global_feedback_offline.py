"""Distinct T5 physical-source reception: authentic ROOT inputs, NumPy/stdlib only.

Component binary identities are recomposed. Aggregate execution association is
ROOT-attested, never cryptographically inferred from filenames or missing payloads.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "180afdc5b294788f370e3720c71772d70805919c"
SCHEMA = "sol61.physical-global-feedback.offline-reception@2"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


math_oracle = load("physical_global_feedback_math", "sol61_integral_feedback_offline_oracle.py")
bounded = load("physical_global_feedback_bounded", "sol61_moving_interval_offline_oracle.py")
require, strict_json, digest = math_oracle.require, math_oracle.strict_json, math_oracle.digest
SOURCE_FILES = (
    "tests/python/integration/runtime/test_public_integral_feedback.py",
    "tests/python/unit/codegen/test_integral_candidate_capture.py",
    "python/pops/codegen/program_global_sources.py",
    "python/pops/codegen/program_emit_model_kernels.py",
    "include/pops/runtime/program/program_context.hpp",
)
IDENTITY_FILES = ("python/pops/identity/artifact.py", "python/pops/identity/digest.py",
                  "python/pops/identity/encoding.py", "python/pops/codegen/_compiled_artifact.py")
ASSOCIATION = "ROOT attested actual case execution/cache association; aggregate payload was not retained"
BASE_KEYS = {"association_scope", "case_artifacts", "extra_sources_and_generated", "identity", "junit",
             "limitations", "mathematical_pins", "mathematical_receipt", "native_build_source", "qualification",
             "schema", "source_commit", "source_files", "status"}
CASE_KEYS = {"aggregate_artifact", "association_authority", "cells", "kind", "program_cpp", "program_sidecar",
             "program_so", "receipt_directory", "system_sidecar", "system_so"}
MARKER = "serialized Program IR (documentary provenance included; excluded from _ir_hash):"


def canonical_path(value):
    require(type(value) is str, "pinned path must be text")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts
            and not any(part.is_symlink() for part in (path, *path.parents))
            and path == path.resolve(strict=True), "pinned path is aliased or noncanonical")
    return path


def leaf(row):
    require(type(row) is dict and set(row) == {"path", "sha256"}
            and type(row["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
            "invalid explicit ROOT file pin")
    path = canonical_path(row["path"])
    raw = bounded.file_bytes(path)
    require(digest(raw) == row["sha256"], "ROOT file bytes changed: " + str(path))
    return path, raw


def identity_data(token, domain):
    require(type(token) is str, "identity token must be text")
    match = re.fullmatch(r"pops\." + re.escape(domain) + r"\.v([1-9][0-9]*):sha256:([0-9a-f]{64})", token)
    require(match is not None, "wrong typed component identity")
    return dict(domain=domain, schema_version=int(match[1]), algorithm="sha256", digest=bytes.fromhex(match[2]))


def binary_cbor(value):
    def head(major, number):
        require(0 <= number < 1 << 64, "CBOR length overflow")
        if number < 24:
            return bytes([32 * major + number])
        for width, marker in ((1, 24), (2, 25), (4, 26), (8, 27)):
            if number < 1 << (8 * width):
                return bytes([32 * major + marker]) + number.to_bytes(width, "big")
        raise ValueError("CBOR overflow")

    if type(value) is bytes:
        return head(2, len(value)) + value
    if type(value) is dict:
        require(all(type(key) is str for key in value), "CBOR keys must be strings")
        rows = sorted(((binary_cbor(key), binary_cbor(item)) for key, item in value.items()),
                      key=lambda row: (len(row[0]), row[0]))
        return head(5, len(rows)) + b"".join(key + item for key, item in rows)
    return math_oracle.cbor(value)


def identity(domain, payload):
    # All values here are bytes/int/string/maps, precisely the production codec's
    # strict value language. No float JSON conversion or PoPS import is involved.
    return "pops.%s.v1:sha256:%s" % (domain, digest(binary_cbor(dict(
        protocol="pops.identity", domain=domain, schema_version=1, payload=payload))))


def component(so_raw, sidecar_raw):
    data = strict_json(sidecar_raw)
    require(type(data) is dict and set(data) == {"artifact_identity", "artifact_spec_identity", "binary_identity",
                                               "protocol", "semantic_identity"}
            and data["protocol"] == "pops.artifact-sidecar.v1", "component sidecar contract differs")
    expected_binary = identity("binary", dict(algorithm="sha256", content_digest=hashlib.sha256(so_raw).digest(),
                                             size=len(so_raw)))
    require(data["binary_identity"] == expected_binary, "component binary identity differs from actual SO")
    spec = identity_data(data["artifact_spec_identity"], "artifact-spec")
    binary = identity_data(data["binary_identity"], "binary")
    semantic = identity_data(data["semantic_identity"], "semantic")
    require(spec["schema_version"] == binary["schema_version"] == 1
            and semantic["schema_version"] == 3, "component identity schema differs")
    require(data["artifact_identity"] == identity("artifact", dict(spec=spec, binary=binary)),
            "component artifact identity does not bind its actual binary")
    return data


def ir_from_cpp(source):
    require(source.count(MARKER) == 1, "missing/duplicate actual serialized Program IR")
    tail = source.split(MARKER, 1)[1].lstrip()
    ir, end = json.JSONDecoder(object_pairs_hook=lambda rows: unique_rows(rows)).raw_decode(tail)
    require(tail[end:].startswith("\n\nlowering provenance"), "Program IR documentary boundary differs")
    require(source.startswith("/*\n") and "*/" in source, "generated CPP provenance boundary differs")
    return ir, source.split("*/", 1)[1]


def unique_rows(rows):
    result = {}
    for key, value in rows:
        require(key not in result, "duplicate embedded Program IR key")
        result[key] = value
    return result


def source_ir(ir):
    require(type(ir) is dict and ir["version"] == 8 and type(ir["version"]) is int
            and ir["name"] == "feedback_lie", "wrong explicit physical-source Program IRv8")
    nodes = ir["nodes"]
    require([row["id"] for row in nodes] == list(range(7))
            and [row["op"] for row in nodes] == ["state", "integral_candidate", "source", "linear_combine",
                                                "pointwise_expression", "rhs", "linear_combine"],
            "physical-source DAG differs")
    point = nodes[0]["point"]
    require(point["offset"] == {"kind": "integer", "value": "0"} and point["step"] == 0
            and point["clock"] == ir["clock"] and nodes[1]["point"] == nodes[2]["point"] == point,
            "candidate capture/source must share exact authored point n")
    attrs = nodes[1]["attrs"]
    require(attrs == dict(capture_version=1, integral="q", scope="candidate",
                          units='{"kind":"physical_dimension","powers":[]}'), "wrong typed integral capture")
    require(nodes[2]["inputs"] == [0, 1] and nodes[2]["attrs"]["source"] == "reaction",
            "physical source does not consume state and unique capture")
    ports = nodes[2]["attrs"]["physical_global_inputs_v1"]
    require(type(ports) is list and len(ports) == 1, "physical global source input inventory differs")
    port = ports[0]
    require(set(port) == {"input", "port", "units", "version"} and port["input"] == 1 and port["version"] == 1
            and port["units"] == attrs["units"], "physical global capture input/units differ")
    handle = port["port"]["handle"]
    require(handle["kind"] == handle["declaration_ref"]["kind"] == "global_quantity"
            and handle["local_id"] == handle["declaration_ref"]["local_id"] == "circuit_quantity"
            and handle["block_ref"] == nodes[0]["block"], "wrong original global declaration/block")
    require(nodes[3]["inputs"] == [2] and nodes[3]["attrs"]["coeffs"] == [[[0, {"kind": "integer", "value": "1"}]]],
            "source balance view must remain a rate with coefficient one")
    require(nodes[4]["inputs"] == [0, 3] and nodes[4]["attrs"]["expressions"] == [4]
            and nodes[4]["attrs"]["expression_nodes"] == [
                ["input", 0, 0], ["coefficient", [[1, {"kind": "integer", "value": "1"}]]],
                ["input", 1, 0], ["mul", 1, 2], ["add", 0, 3]],
            "reaction candidate contains an extra capture or duration multiplier")
    require(nodes[5]["inputs"] == [4] and nodes[6]["inputs"] == [4, 5]
            and len(ir["commits"]) == 1 and ir["commits"][0]["value"] == 6,
            "transport or publication source differs")
    require(ir["external_trace_transfers"] == [dict(axis=0, component=0, rate=5, scale=-1., side=1, state="q")],
            "external trace occurrence/selection differs")
    require(ir["integral_states"] == [dict(initial=0.7, name="q")], "original integral initial value differs")
    return dict(version=8, capture_node=1, source_node=2, source_point=point,
                circuit_port=handle["qualified_id"], reaction_expression="-0.3*circuit_quantity*density")


def cpp_source(code, quantity):
    # Scoped executable body, not the inert documentary text or a global substring match.
    begin = code.index('const auto _pt2 =')
    end = code.index('ctx.profile_record("node:reaction", _pt2);', begin)
    body = code[begin:end]
    require(body.count("ctx.integral_candidate_value(") == 1, "physical capture must be evaluated once")
    call = re.search(r'const pops::Real (physical_global_2_1) = ctx\.integral_candidate_value\(integral_capture_1,"([^"\n]+)",', body)
    require(call is not None and call[2] == quantity, "physical source uses foreign candidate quantity")
    require(call.start() < body.index("for (int li =") < body.index("pops::for_each_cell("),
            "physical capture is not outside patch/cell loops")
    expressions = re.findall(r"outA\(index, 0\) = ([^;]+);", body)
    expected = ast.dump(ast.parse("(-0.3 * physical_global_2_1) * mass", mode="eval"))
    require(len(expressions) == 1 and ast.dump(ast.parse(expressions[0], mode="eval")) == expected,
            "compiled original physical reaction is wrong or multiplied by q twice")
    require(re.findall(r"const pops::Real mass = ([^;]+);", body) == ["u0A(index, 0)"],
            "compiled source reads a foreign state")
    require(code.count('ctx.capture_integral_candidate("' + quantity + '",') == 1,
            "compiled candidate capture inventory differs")
    require('ctx.axpy(u3, static_cast<pops::Real>(pops::Real(1)), r2, dt, {{0, 1, 1}});' in code,
            "balance helper coefficient differs")
    begin = code.index('const auto _pt4 =')
    end = code.index('ctx.profile_record("node:reaction_candidate", _pt4);', begin)
    candidate = code[begin:end]
    expected_rows = {"cse0_": "u0A(index, 0)", "cse1_": "(static_cast<pops::Real>(dt))",
                     "cse2_": "u3A(index, 0)", "cse3_": "(cse1_ * cse2_)", "cse4_": "(cse0_ + cse3_)"}
    actual_rows = dict(re.findall(r"const pops::Real (cse[0-9]+_) = ([^;]+);", candidate))
    require(actual_rows == expected_rows and re.findall(r"outA\(index, 0\) = ([^;]+);", candidate) ==
            ["u0A(index, 0)", "cse4_"], "compiled candidate rate/duration composition differs")
    return dict(capture_outside_cell_loop=True, physical_capture_evaluations=1, extra_q_multiplier=False,
                balance_rate_coefficient=1, candidate_dt_multipliers=1)


def source_authority(owner, extras, catalog):
    verified = {}
    for relative in SOURCE_FILES + IDENTITY_FILES:
        frozen = subprocess.check_output(["git", "-C", str(ROOT), "show", SOURCE + ":" + relative])
        if not relative.startswith("tests/"):
            require(catalog[relative] == digest(frozen), "ROOT production source catalogue differs from frozen Git")
        if relative in SOURCE_FILES:
            matches = [row for row in extras if row["path"].endswith("/" + relative)]
            require(len(matches) == 1, "missing/duplicate explicit scientific source leaf")
            _, raw = leaf(matches[0])
            require(raw == frozen, "actual scientific source differs from frozen Git")
        verified[relative] = digest(frozen)
    context = subprocess.check_output(["git", "-C", str(ROOT), "show", SOURCE + ":" + SOURCE_FILES[-1]], text=True)
    overload = context.split("void axpy(field_type& destination, Real factor, const field_type& source, Real,", 1)[1].split("\n  }", 1)[0]
    require(overload == "\n            std::initializer_list<ExactCoefficientTerm>) const {\n    axpy(destination, factor, source);",
            "context overload applies an extra duration")
    require(owner["source_commit"] == SOURCE, "wrong actual source contract")
    return verified


def junit(raw, rank, size, associations):
    require(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "unsafe JUnit declarations")
    root = ET.fromstring(raw)
    cases = list(root.iter("testcase"))
    require(cases and all(not list(case.findall("failure")) and not list(case.findall("error"))
                         and not list(case.findall("skipped")) for case in cases), "native JUnit has unsuccessful cases")
    for suite in root.iter("testsuite"):
        require(int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == int(suite.attrib["skipped"]) == 0
                and int(suite.attrib["tests"]) == len(list(suite.iter("testcase"))), "JUnit aggregate differs")
    selected = {}
    for kind, cells in (("restart", 8), ("restart", 16), ("retry", 8)):
        name = ("test_public_feedback_original_source_real_exterior_and_byte_exact_restart[True-%d]" % cells
                if kind == "restart" else "test_feedback_rejected_transport_preserves_capture_integral_and_safe_retry[True]")
        matches = [case for case in cases if case.get("name") == name]
        require(len(matches) == 1, "missing/duplicate authentic physicalTrue JUnit witness")
        props = unique_rows([(row.get("name"), row.get("value")) for row in matches[0].findall("properties/property")])
        association = associations[(kind, cells)]
        if kind == "restart":
            require(props["physical_global_source"] == "True" and props["dim"] == "2"
                    and props["rank"] == str(rank) and props["size"] == str(size)
                    and props["artifact_identity"] == association["aggregate_artifact"]
                    and props["feedback_receipts"] == association["receipt_directory"], "True witness association differs")
        selected[name] = dict(properties=props, retry_directory_association="ROOT-attested" if kind == "retry" else "JUnit")
    return dict(rank=rank, successful_native_cases=len(cases), physicalTrue_cases=selected)


def receive(path, owner_sha):
    require(type(owner_sha) is str and re.fullmatch("[0-9a-f]{64}", owner_sha), "external ROOT seal required")
    raw = bounded.file_bytes(canonical_path(str(path)))
    require(digest(raw) == owner_sha, "external ROOT seal differs")
    owner = strict_json(raw)
    require(owner["schema"] == "pops.root.physical-global-reception-inputs@1"
            and owner["status"] == "pending_independent_source_linkage", "ROOT input contract differs")
    math_path, math_raw = leaf(owner["mathematical_pins"])
    pins = strict_json(math_raw)
    size = pins["size"]
    require(type(size) is int and size in (1, 2), "unsupported witness rank count")
    require(set(owner) == BASE_KEYS | ({"launcher_result", "rank_junits", "rank_native_identities"} if size == 2 else set()),
            "ROOT inputs inventory differs")
    _, identity_raw = leaf(owner["identity"])
    runtime = strict_json(identity_raw)
    require(runtime["source_commit"] == pins["source_commit"] == owner["source_commit"] == SOURCE
            and runtime["native_sha256"] == pins["native_sha256"] and runtime["abi_key"] == pins["abi_key"],
            "actual source/native/SDK runtime identity differs")
    _, catalog_raw = leaf(owner["source_files"])
    catalog = strict_json(catalog_raw)
    require(len(catalog) == 1114 and digest(catalog_raw) == runtime["source_files_sha256"],
            "actual attested production catalogue differs")
    extras = owner["extra_sources_and_generated"]
    require(type(extras) is list and len(extras) == 20
            and len({row["path"] for row in extras}) == 20, "actual source/generated inventory differs")
    extra_bytes = {row["path"]: leaf(row)[1] for row in extras}
    source_pins = source_authority(owner, extras, catalog)
    associations = owner["case_artifacts"]
    require(type(associations) is list and len(associations) == 3
            and sorted((row["kind"], row["cells"]) for row in associations) == [("restart", 8), ("restart", 16), ("retry", 8)],
            "actual True case association inventory differs")
    math_cases = {(row["kind"], row["cells"]): row for row in pins["cases"]}
    source_results = []
    for row in associations:
        require(set(row) == CASE_KEYS and row["association_authority"] == ASSOCIATION, "unattested aggregate association")
        case = math_cases[(row["kind"], row["cells"])]
        require(row["aggregate_artifact"] == case["artifact"], "associated aggregate is foreign")
        directory = canonical_path(row["receipt_directory"])
        require(set(case["phases"]) == set(math_oracle.PHASES[row["kind"]]), "wrong authentic phase set")
        for phase in case["phases"].values():
            for pin in phase.values():
                require(canonical_path(pin["path"]).parent == directory, "phase escapes owner-attested execution directory")
        component_bytes = {}
        for kind in ("program_cpp", "program_so", "program_sidecar", "system_so", "system_sidecar"):
            pin = row[kind]
            require(pin in extras and canonical_path(pin["path"]).is_relative_to(directory),
                    "associated component is not an explicit executed-case ROOT leaf")
            component_bytes[kind] = extra_bytes[pin["path"]]
        program = component(component_bytes["program_so"], component_bytes["program_sidecar"])
        system = component(component_bytes["system_so"], component_bytes["system_sidecar"])
        source = component_bytes["program_cpp"].decode("utf-8")
        ir, code = ir_from_cpp(source)
        ir_evidence = source_ir(ir)
        cpp_evidence = cpp_source(code, case["quantity_identity"])
        require(re.findall(r"^program_hash\s+: ([0-9a-f]{64})$", source, re.MULTILINE) ==
                [program["semantic_identity"].rsplit(":", 1)[1]], "CPP semantic identity differs from actual Program sidecar")
        source_results.append(dict(kind=row["kind"], cells=row["cells"], aggregate_artifact=case["artifact"],
                                   aggregate_binding="ROOT-attested execution association; payload not retained",
                                   program=program, system=system, ir=ir_evidence, cpp=cpp_evidence))
    associated = {(row["kind"], row["cells"]): row for row in associations}
    junit_rows = owner["rank_junits"] if size == 2 else [owner["junit"]]
    require(len(junit_rows) == size and junit_rows[0] == owner["junit"], "all-rank JUnit inventory differs")
    native_results = [junit(leaf(row)[1], rank, size, associated) for rank, row in enumerate(junit_rows)]
    if size == 2:
        require(len(owner["rank_native_identities"]) == 2, "rank runtime inventory differs")
        for rank, row in enumerate(owner["rank_native_identities"]):
            rank_identity = strict_json(leaf(row)[1])
            require(set(rank_identity) == {"rank", "ranks", "dimension", "package_file", "native_file",
                                          "native_sha256", "execution_environment"}
                    and type(rank_identity["rank"]) is int and rank_identity["rank"] == rank
                    and rank_identity["ranks"] == 2 and rank_identity["dimension"] == 2
                    and rank_identity["native_sha256"] == pins["native_sha256"]
                    and rank_identity["package_file"] == runtime["package_file"]
                    and rank_identity["native_file"] == runtime["native_file"], "rank native execution differs")
        launcher = strict_json(leaf(owner["launcher_result"])[1])
        require(launcher["schema_version"] == 3 and launcher["status"] == "passed"
                and launcher["returncode"] == 0 and launcher["timeout"] is False
                and launcher["ranks"] == launcher["dimension"] == 2
                and launcher["same_installation"] is True and launcher["test_sources_unchanged"] is True
                and launcher["rank_test_parity"] is True
                and launcher["authentication_before"] == launcher["authentication_after"] == 0,
                "native MPI launcher authentication differs")
    leaf(owner["mathematical_receipt"])
    math_result = math_oracle.receive(math_path)
    return dict(schema=SCHEMA, status="received", physical_global_source=True, owner_sha256=owner_sha,
                source_commit=SOURCE, native_build_source=owner["native_build_source"],
                native_sha256=pins["native_sha256"], abi_key=pins["abi_key"], ranks=size,
                qualification="mathematical receipt + actual physical-source/CPP/component byte authentication + ROOT execution association",
                cryptographic_aggregate_binding=False, component_binary_binding_recomputed=True,
                native_execution_here=False, source_pins=source_pins, native_junit=native_results,
                cases=source_results, mathematical_reception=math_result,
                historical_source_contract_upcast=False,
                limits=["aggregate original payload was not retained; association is ROOT-attested",
                        "offline receipt only; no new native/MPI execution or AMR/vector/3D qualification"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-inputs", type=Path, required=True)
    parser.add_argument("--owner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = receive(args.root_inputs, args.owner_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key in {"schema", "status", "ranks", "owner_sha256", "cryptographic_aggregate_binding"}}))


if __name__ == "__main__":
    main()
