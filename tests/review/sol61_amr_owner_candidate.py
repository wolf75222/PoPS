"""Prepare inventories for ROOT audit; never seal, approve or qualify a dataset."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


existing = load("amr_existing_inventory", "sol61_m18_owner_assemble.py")
reader = load("amr_candidate_reader", "sol61_evolved_stage_amr_saved_reception_v2.py")
need = existing.require


def assemble(spec):
    roots = [existing.root_directory(path) for path in spec["roots"]]
    def pin(path):
        path = Path(path)
        matching = [root for root in roots if path.is_relative_to(root)]
        need(len(matching) == 1, "input must have one explicit nonoverlapping root")
        return existing.leaf(path, matching[0])
    def read_pin(row):
        need(pin(row["path"]) == row, "receipt pin changed or resealed input differs")
        return Path(row["path"]).read_bytes()
    ranks = 1 if spec["mode"] == "serial" else 2 if spec["mode"] == "mpi2" else 0
    need(ranks > 0 and spec["ir_version"] in (16, 17), "unsupported mode/IR")
    for key in ("source_commit", "native_build_source_commit"):
        need(re.fullmatch("[0-9a-f]{40}", spec[key]) is not None, "exact commit required")
    identities = {}
    for phase in ("before", "after"):
        identities[phase] = existing.strict_json(read_pin(pin(spec["identity_"+phase])))
        need(set(identities[phase]) == {"native", "sdk", "package_manifest"}, "identity inventory must name exact three origins")
        for row in identities[phase].values():
            read_pin(row)
    need(identities["before"] == identities["after"], "native/package/SDK identity changed around execution")
    origins = identities["before"]
    source_files = {key: pin(path) for key, path in spec["source_files"].items()}
    need(set(source_files) == {"amr", "equations", "controls", "fixture"}, "four actual source files required")
    reader.declared_source(*(read_pin(source_files[key]) for key in ("amr", "equations", "controls")))
    cases, seen_cases, names_by_rank, tests_by_rank, junit = [], set(), {}, {}, []
    for row in spec["junit"]:
        rank = row["rank"]
        need(type(rank) is int and 0 <= rank < ranks and rank not in names_by_rank, "duplicate/foreign JUnit rank")
        leaf = pin(row["path"]); raw = read_pin(leaf)
        need(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "external XML declarations forbidden")
        tree = ET.fromstring(raw); tests = tree.findall(".//testcase")
        need(not any(node.tag in ("failure", "error", "skipped") for node in tree.iter()), "raw batch has failure/error/skip")
        for suite in tree.iter("testsuite"):
            need(all(int(suite.attrib.get(k, "0")) == 0 for k in ("failures", "errors", "skipped")), "raw suite is not clean")
        names = [test.attrib["name"] for test in tests]
        need(len(names) == len(set(names)), "duplicate JUnit test name")
        names_by_rank[rank] = names
        tests_by_rank[rank] = {test.attrib["name"]:test for test in tests}
        junit.append(dict(rank=rank, file=leaf))
    need(set(names_by_rank) == set(range(ranks)), "missing JUnit rank")
    need(all(sorted(names) == sorted(names_by_rank[0]) for names in names_by_rank.values()), "rank batch inventory mismatch")
    for path in spec["receipts"]:
        receipt_pin = pin(path); receipt = existing.strict_json(read_pin(receipt_pin))
        key = (receipt["cells"], receipt["width"])
        need(key in reader.CASES and key not in seen_cases, "duplicate/foreign receipt case")
        seen_cases.add(key)
        need(receipt["fixture_schema"] == "pops.evolved-stage-amr-native-fixture@2"
             and (receipt["rank"], receipt["size"], receipt["dimension"]) == (0, ranks, 2), "receipt runtime rank/dimension differs")
        need(reader.typed(receipt["newton"], reader.CONTROLS) and receipt["dt"] == reader.DT
             and receipt["fd_step"] == 1e-6 and receipt["acceptance"] == reader.TOL, "receipt controls differ")
        files, origins_seen = {}, set()
        def retain(row):
            raw = read_pin(row)
            need(row["path"] not in origins_seen, "aliased receipt file roles")
            origins_seen.add(row["path"]); files[row["path"]] = row["sha256"]
            return raw
        need(receipt["native"] == origins["native"], "receipt DSO differs from before/after identity")
        retain(receipt["native"])
        registry = existing.strict_json(retain(receipt["carrier_registry"]))
        need(set(registry) == {"schema", "dimension", "size", "phases"}
             and registry["schema"] == "sol61.amr.carrier-registry@1"
             and registry["dimension"] == 2 and registry["size"] == ranks
             and set(registry["phases"]) == set(reader.PHASES), "carrier inventory schema differs")
        for phase in reader.PHASES:
            row = registry["phases"][phase]
            need(set(row) == {"rows_by_rank"} and type(row["rows_by_rank"]) is list
                 and len(row["rows_by_rank"]) == ranks, "carrier inventory rank missing")
        for phase in reader.PHASES:
            for row in receipt["phases"][phase]["levels"]:
                retain(row)
        checkpoints = {phase: reader.wire.archive(retain(receipt["checkpoints"][phase]))
                       for phase in ("accepted", "continuous", "replay")}
        manifest = existing.strict_json(str(checkpoints["accepted"]["pops_checkpoint_manifest"].item()))
        ids = {key: reader.wire.identity_token(manifest[key+"_identity"], key)
               for key in ("artifact", "bind", "semantic")}
        need(ids["artifact"] == receipt["artifact"], "receipt artifact differs from CP")
        for arrays in checkpoints.values():
            reader.envelope(arrays, ids["artifact"], ids["bind"], ids["semantic"])
            need(int(arrays["pops_amr_checkpoint_version"]) == 12, "historical CP cannot enter candidate")
            reader.carriers.decode(arrays["state_carriers_checkpoint"])
        programs = 0
        for component in receipt["compilation"]:
            retain(component["DSO"]); retain(component["sidecar"])
            if component["component"].startswith("program-"):
                programs += 1
                ir = existing.strict_json(retain(component["ir.json"]))
                cpp = retain(component["cpp"]).decode()
                reader.program_image(ir, cpp, spec["ir_version"], component["program_hash"], key[1])
        need(programs == 1, "one retained actual CPP/IR program required")
        name = f"test_public_evolved_stage_amr_checkpoint_and_composite_Q[{key[1]}-{key[0]}]"
        need(name in names_by_rank[0], "receipt missing from raw JUnit batch")
        for rank, tests in tests_by_rank.items():
            properties = tests[name].findall("./properties/property")
            props = {row.attrib["name"]:row.attrib["value"] for row in properties}
            need(len(props) == len(properties), "duplicate JUnit property")
            need(props.get("rank") == str(rank) and props.get("size") == str(ranks)
                 and props.get("dimension") == "2" and props.get("artifact_identity") == ids["artifact"]
                 and props.get("evolved_stage_amr_receipt") == receipt_pin["path"], "JUnit rank/receipt identity differs")
        # This assembly does not assert scientific reception or write an approval.
        cases.append(dict(cells=key[0], width=key[1], receipt=receipt_pin, files=files, **ids))
    need(seen_cases == reader.CASES, "all four completed actual cases required")
    pins = dict(schema="sol61.evolved-stage-amr.owner-pins@2", qualification=reader.QUALIFICATION,
        native_evidence=True, mode=spec["mode"], ranks=ranks, source_commit=spec["source_commit"],
        native_build_source_commit=spec["native_build_source_commit"], abi_key=spec["abi_key"],
        ir_version=spec["ir_version"], roots=list(map(str, roots)), source_files=source_files,
        junit=junit, batch_names=names_by_rank[0], cases=cases, **origins)
    return dict(status="candidate_pending_ROOT_audit", scientific_reception=False,
        owner_pins_candidate=pins, native_build_receipt=pin(spec["native_build_receipt"]),
        identity_before=pin(spec["identity_before"]), identity_after=pin(spec["identity_after"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec"); parser.add_argument("candidate")
    args = parser.parse_args()
    result = assemble(existing.strict_json(Path(args.spec).read_bytes()))
    with Path(args.candidate).open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
