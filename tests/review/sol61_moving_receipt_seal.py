"""Assemble real ALE donor paths from owner-pinned JUnit; never run PoPS.

The output is a pending reception inventory, not a physical PASS. The external
owner seals its generated pins and runs sol61_moving_interval_offline_oracle.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import sol61_moving_interval_offline_oracle as oracle

require, strict_json, digest = oracle.require, oracle.strict_json, oracle.digest
SCHEMA = "sol61.moving-interval.launch-owner-pins@1"
CLASS = "tests.python.integration.runtime.test_public_moving_interval"


def donor(launch, path):
    value = Path(path)
    require(value.is_absolute(), "donor path must be explicit and absolute")
    value = value.resolve(strict=True)
    require(value.is_file() and value.is_relative_to(launch), "donor escapes launch directory")
    return value


def pin(launch, path):
    value = donor(launch, path)
    return dict(path=str(value), sha256=digest(value.read_bytes()))


def expected_nodes():
    return {"test_declared_moving_public_chain[%s-components%d]" % (source, component)
            for source in (False, True) for component in range(3)} | {
        "test_rejected_moving_interval_preserves_state_geometry_wire_and_safe_retry[%d]" % n
        for n in (16, 32)}


def junit_cases(launch, path, rank, size):
    root = ET.fromstring(donor(launch, path).read_bytes())
    nodes = list(root.iter("testcase"))
    require(len(nodes) == 8 and {row.get("name") for row in nodes} == expected_nodes(),
            "JUnit lacks the exact eight ALE nodes")
    suites = list(root.iter("testsuite"))
    require(suites and all(int(row.get(key, "0")) == 0 for row in suites
                          for key in ("failures", "errors", "skipped")), "JUnit lane is incomplete")
    cases, inventories = [], []
    for row in nodes:
        require(row.get("classname") == CLASS and not any(row.find(tag) is not None
                for tag in ("failure", "error", "skipped")), "JUnit node did not pass")
        entries = [(value.get("name"), value.get("value")) for value in row.findall("./properties/property")]
        require(len(dict(entries)) == len(entries), "duplicate JUnit property")
        props = dict(entries)
        require(props.get("rank") == str(rank) and props.get("size") == str(size)
                and props.get("dimension") == "1", "JUnit rank/size/dimension differs")
        inventory = donor(launch, props["moving_receipts"])
        payload = strict_json(inventory.read_bytes())
        require(type(payload) is list and len(payload) == (2 if "chain[" in row.get("name") else 1),
                "wrong case count in moving_receipts")
        actual_artifacts = {value["artifact"] for value in payload}
        named = props["artifact_identity"]
        require(actual_artifacts == set(strict_json(named) if named.startswith("[") else [named]),
                "JUnit artifact identities differ from moving_receipts")
        for value in payload:
            require(type(value["dimension"]) is int and value["dimension"] == 1
                    and type(value["size"]) is int and value["size"] == size
                    and type(value["cells"]) is int and type(value["source"]) is bool
                    and type(value["components"]) is list
                    and all(type(name) is str for name in value["components"]),
                    "case dimension/size/source/components differs")
            if value["kind"] == "restart":
                require(props["artifact_identity_cells_%d" % value["cells"]] == value["artifact"]
                        and props["dimension_cells_%d" % value["cells"]] == "1",
                        "per-resolution artifact property differs")
            cases.append((value, inventory))
        inventories.append(pin(launch, inventory))
    labels = [(row["kind"], row["cells"], tuple(row["components"]), row["source"]) for row, _ in cases]
    require(len(labels) == 14 and len(set(labels)) == 14 and set(labels) == set(oracle.campaign()),
            "incomplete/duplicate physical case campaign")
    return cases, inventories


def cbor_bytes(value):
    # Independent extension of the oracle's narrow CBOR for binary digests.
    if type(value) is bytes:
        require(len(value) == 32, "binary digest must be 32 bytes")
        return b"\x58\x20" + value
    if type(value) is dict:
        rows = sorted(((oracle.common.cbor(key), cbor_bytes(item)) for key, item in value.items()),
                      key=lambda row: (len(row[0]), row[0]))
        require(len(rows) < 24, "identity envelope too large")
        return bytes([0xa0 + len(rows)]) + b"".join(key + item for key, item in rows)
    return oracle.common.cbor(value)


def identity_data(token, domain):
    match = re.fullmatch(r"pops\." + re.escape(domain) + r"\.v1:sha256:([0-9a-f]{64})", token)
    require(match is not None, "unknown sidecar identity domain/version")
    return dict(domain=domain, schema_version=1, algorithm="sha256", digest=bytes.fromhex(match[1]))


def identity_token(domain, payload):
    envelope = dict(protocol="pops.identity", domain=domain, schema_version=1, payload=payload)
    return "pops.%s.v1:sha256:%s" % (domain, digest(cbor_bytes(envelope)))


def compiled_files(launch, inventories):
    rows = []
    caches = {Path(row["path"]).parent / "pops-native-cache" for row in inventories}
    for cache in sorted(caches):
        sidecars = sorted(cache.glob("*.so.pops-artifact.json"))
        require(sidecars, "missing real compile artifact sidecars")
        require({str(value) for value in cache.glob("*.so")} ==
                {str(value).removesuffix(".pops-artifact.json") for value in sidecars},
                "DSO/sidecar inventory differs")
        for sidecar in sidecars:
            data = strict_json(donor(launch, sidecar).read_bytes())
            require(set(data) == {"protocol", "semantic_identity", "artifact_spec_identity",
                                  "binary_identity", "artifact_identity"}
                    and data["protocol"] == "pops.artifact-sidecar.v1", "unknown compile sidecar")
            binary = donor(launch, str(sidecar).removesuffix(".pops-artifact.json"))
            raw = binary.read_bytes()
            computed = identity_token("binary", dict(algorithm="sha256",
                                       content_digest=hashlib.sha256(raw).digest(), size=len(raw)))
            require(data["binary_identity"] == computed, "DSO bytes differ from compile identity")
            spec = identity_data(data["artifact_spec_identity"], "artifact-spec")
            emitted = identity_data(computed, "binary")
            require(data["artifact_identity"] == identity_token("artifact", dict(spec=spec, binary=emitted)),
                    "compile artifact identity does not bind specification and DSO")
            require(re.fullmatch(r"pops.semantic.v[1-9][0-9]*:sha256:[0-9a-f]{64}",
                                data["semantic_identity"]) is not None, "invalid compile semantic identity")
            rows.append(dict(sidecar=pin(launch, sidecar), dso=pin(launch, binary), identities=data))
    return rows


def template(launch, rank):
    identity = launch / "identity.json"
    if not identity.is_file():
        identity = launch / "before/identity.json"
    data = strict_json(donor(launch, identity).read_bytes())
    junit = launch / ("pytest.xml" if (launch / "pytest.xml").is_file() else "rank%d.xml" % rank)
    root = ET.fromstring(donor(launch, junit).read_bytes())
    sizes = {row.get("value") for row in root.findall(".//property[@name='size']")}
    require(len(sizes) == 1, "JUnit size authority is absent or ambiguous")
    return dict(schema=SCHEMA, authority="pending_external_owner_approval", rank=rank, size=int(sizes.pop()),
                dimension=1, source_commit=data["source_commit"], native_sha256=data["native_sha256"],
                abi_key=data["abi_key"], identity_file=pin(launch, identity), junit_file=pin(launch, junit),
                source_files_file=pin(launch, identity.parent / "source-files.json"),
                oracle_contract=oracle.contract())


def assemble(launch, rank, owner):
    require(owner["schema"] == SCHEMA and owner["authority"] == "external_owner"
            and type(owner["rank"]) is int and owner["rank"] == rank
            and type(owner["size"]) is int and 0 <= rank < owner["size"]
            and type(owner["dimension"]) is int and owner["dimension"] == 1,
            "missing external owner/rank/dimension authority")
    for key in ("identity_file", "junit_file", "source_files_file"):
        value = donor(launch, owner[key]["path"])
        require(pin(launch, value) == owner[key], "external owner file SHA256 differs: " + key)
    identity = strict_json(Path(owner["identity_file"]["path"]).read_bytes())
    require(all(identity[key] == owner[key] for key in ("source_commit", "native_sha256", "abi_key"))
            and re.fullmatch("[0-9a-f]{40}", owner["source_commit"])
            and re.fullmatch("[0-9a-f]{64}", owner["native_sha256"])
            and "dim=1" in owner["abi_key"], "owner source/native/ABI identity differs")
    require(identity["source_diff_sha256"] == digest(b""), "launch source was dirty")
    sources = strict_json(Path(owner["source_files_file"]["path"]).read_bytes())
    require(identity["source_files_sha256"] == owner["source_files_file"]["sha256"]
            and identity["verified_source_files"] == len(sources)
            and all(type(key) is str and re.fullmatch("[0-9a-f]{64}", value) for key, value in sources.items()),
            "installed source inventory differs")
    actual, inventories = junit_cases(launch, owner["junit_file"]["path"], rank, owner["size"])
    cases = []
    for row, inventory in actual:
        require(tuple(row["phases"]) == oracle.PHASES[row["kind"]], "phase inventory differs")
        directory = (Path(row["checkpoint"]).parent if row["kind"] == "restart" else inventory.parent)
        phases, platform = {}, None
        for phase in row["phases"]:
            receipt_path = donor(launch, directory / (phase + "-receipt.json"))
            receipt = strict_json(receipt_path.read_bytes())
            require(receipt["phase"] == phase and receipt["artifact"] == row["artifact"]
                    and all(type(receipt[key]) is int for key in
                            ("rank", "size", "dimension", "macro_step", "generation"))
                    and type(receipt["time"]) is float
                    and receipt["rank"] == 0 and receipt["size"] == owner["size"]
                    and receipt["dimension"] == 1 and receipt["moving_identity"] == row["moving_identity"]
                    and receipt["physical_frame"] == row["physical_frame"], "phase receipt authority differs")
            platform = receipt["platform"] if platform is None else platform
            require(receipt["platform"] == platform, "platform changes between phases")
            checkpoint = pin(launch, receipt["checkpoint"])
            require(checkpoint["sha256"] == receipt["checkpoint_sha256"], "receipt checkpoint SHA256 differs")
            phases[phase] = dict(receipt=pin(launch, receipt_path), checkpoint=checkpoint,
                                 state=pin(launch, directory / (phase + "-state.npz")))
        expected_outputs = {"step1", "accepted", "continuous", "replayed"} if row["kind"] == "restart" else {"retried"}
        require(set(row["scientific_outputs"]) == expected_outputs, "scientific NPZ inventory differs")
        cases.append({**{key: row[key] for key in ("kind", "cells", "components", "source", "velocity",
                      "artifact", "moving_identity", "physical_frame")}, "platform": platform, "phases": phases,
                      "outputs": {phase: pin(launch, path) for phase, path in row["scientific_outputs"].items()}})
    pins = dict(schema=oracle.SCHEMA, dimension=1, size=owner["size"], cases=cases,
                **{key: owner[key] for key in ("source_commit", "native_sha256", "abi_key", "identity_file")})
    compiled = compiled_files(launch, inventories)
    report = dict(schema="sol61.moving-interval.sealed-inventory@1", scientific_status="pending_owner_reception",
                  cases=len(cases), snapshots=sum(len(row["phases"]) for row in cases),
                  scientific_npz=sum(len(row["outputs"]) for row in cases), rank=rank, size=owner["size"],
                  launch_dir=str(launch), junit=owner["junit_file"], source_files=owner["source_files_file"],
                  moving_receipts=inventories, compiled_artifacts=compiled,
                  generated_source_receipts="unavailable: emitted .cpp/commands were not retained by this launch",
                  public_bundle_component_link="not recorded in moving_receipts; no inference from cache filenames",
                  owner_next_step="Seal oracle-pins.json externally; invoke the genuine offline oracle without changing its guards.")
    return pins, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--owner-pins", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    launch = args.launch_dir.resolve(strict=True)
    require(args.rank >= 0 and launch.is_dir(), "invalid launch/rank")
    if args.template:
        print(json.dumps(template(launch, args.rank), indent=2))
        return
    require(args.owner_pins is not None and args.output_dir is not None, "owner pins and fresh output are required")
    output = args.output_dir.resolve()
    require(not output.exists() and not output.is_relative_to(launch), "output must be fresh and outside donor launch")
    pins, report = assemble(launch, args.rank, strict_json(args.owner_pins.read_bytes()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "oracle-pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output_dir=str(output), scientific_status=report["scientific_status"],
                          cases=report["cases"], snapshots=report["snapshots"], scientific_npz=report["scientific_npz"])))


if __name__ == "__main__":
    main()
