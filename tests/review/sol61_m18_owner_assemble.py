"""Assemble pending actual M18 file pins; ROOT approval precedes scientific reception.

This stdlib/NumPy offline tool never imports PoPS or writes native/scientific data.
The separate entropy oracle performs scientific reception after external sealing.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

_spec = importlib.util.spec_from_file_location(
    "m18_owner_existing_oracle", Path(__file__).with_name("sol61_m18_entropy_offline_oracle.py")
)
oracle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oracle
_spec.loader.exec_module(oracle)
require, digest, strict_json = oracle.require, oracle.digest, oracle.strict_json


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def root_directory(path):
    path = Path(path)
    require(
        path.is_absolute() and path == path.resolve() and path.is_dir(),
        "root must be an existing canonical directory",
    )
    return path


def contained(path, root):
    path, root = Path(path), root_directory(root)
    require(
        path.is_absolute() and path == path.resolve() and path.is_relative_to(root),
        "path escapes or aliases its explicit root",
    )
    require(
        path.is_file() and not path.is_symlink(),
        "missing/nonregular input file",
    )
    return path


def leaf(path, root):
    path = contained(path, root)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"path": str(path), "sha256": hasher.hexdigest()}


def verify_leaf(row, root):
    require(
        type(row) is dict
        and set(row) == {"path", "sha256"}
        and type(row["path"]) is str
        and type(row["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", row["sha256"]),
        "malformed file identity",
    )
    actual = leaf(row["path"], root)
    require(actual == row, "input file digest changed")
    return Path(row["path"])


def strict_junit(raw, pins, rank, data_directory):
    require(
        b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "JUnit external declarations forbidden"
    )
    tree = ET.fromstring(raw)
    cases = list(tree.iter("testcase"))
    require(
        len(cases) == 1 and cases[0].attrib.get("name") == oracle.TEST_NAME,
        "JUnit must contain exactly one M18 witness",
    )
    require(
        not any(node.tag in {"failure", "error", "skipped"} for node in tree.iter()),
        "JUnit has failure/error/skip",
    )
    for suite in (node for node in tree.iter() if node.tag in {"testsuites", "testsuite"}):
        require(
            (
                suite.attrib.get("tests") == "1"
                if suite.tag == "testsuite"
                else suite.attrib.get("tests", "1") == "1"
            )
            and all(suite.attrib.get(key, "0") == "0" for key in ("failures", "errors", "skipped")),
            "JUnit aggregate counts differ",
        )
    properties = list(cases[0].iter("property"))
    names = [row.attrib.get("name") for row in properties]
    require(
        all(type(name) is str and name for name in names) and len(set(names)) == len(names),
        "JUnit duplicate/malformed properties",
    )
    oracle.check_junit(raw, pins, rank)
    values = {row.attrib["name"]: row.attrib["value"] for row in properties}
    require(
        values.get("saved_receipts") == str(data_directory), "JUnit saved receipt realm differs"
    )
    require(
        all("outside_cone_failure_%d" % attempt in values for attempt in (0, 1)),
        "JUnit outside refusals are absent",
    )
    return values


def phase_inventory(directory, archive_root):
    directory = root_directory(directory)
    require(
        directory.is_relative_to(root_directory(archive_root)),
        "data directory escapes archive root",
    )
    phases, expected = {}, {directory / "provenance.json"}
    for phase in oracle.PHASES:
        receipt_path, state_path = (
            directory / (phase + "-receipt.json"),
            directory / (phase + "-state.npz"),
        )
        receipt = strict_json(contained(receipt_path, archive_root).read_bytes())
        require(receipt.get("phase") == phase, "foreign phase receipt")
        name = receipt.get("checkpoint")
        require(
            type(name) is str
            and name
            and Path(name).name == name
            and name not in {".", ".."}
            and "\\" not in name,
            "checkpoint path escape",
        )
        checkpoint = directory / name
        require(
            checkpoint not in expected and checkpoint not in {state_path, receipt_path},
            "checkpoint reused or aliases another file",
        )
        require(
            digest(contained(checkpoint, archive_root).read_bytes())
            == receipt.get("checkpoint_sha256"),
            "checkpoint receipt digest differs",
        )
        phases[phase] = {
            "state": leaf(state_path, archive_root),
            "receipt": leaf(receipt_path, archive_root),
            "checkpoint": leaf(checkpoint, archive_root),
        }
        expected.update((state_path, receipt_path, checkpoint))
    entries = set(directory.iterdir())
    require(
        entries == expected
        and all(entry.is_file() and not entry.is_symlink() for entry in entries),
        "data file inventory is not closed",
    )
    require(len(expected) == 31, "phase inventory is not ten complete triples plus provenance")
    return phases


def origin_metadata(path, roots, provenance):
    path = contained(path, roots["archive"])
    data = strict_json(path.read_bytes())
    require(
        type(data) is dict
        and set(data)
        == {
            "schema",
            "source_commit",
            "python_package",
            "sdk",
            "native",
            "system_packages",
            "generated_cpp",
        }
        and data["schema"] == "sol61.m18-execution-owner-metadata@1",
        "execution metadata contract differs",
    )
    require(
        type(data["source_commit"]) is str and re.fullmatch("[0-9a-f]{40}", data["source_commit"]),
        "invalid source commit",
    )
    for kind in ("python_package", "sdk", "native"):
        verify_leaf(data[kind], roots["installation"])
    require(
        type(data["system_packages"]) is dict
        and set(data["system_packages"]) == {"dual", "target"},
        "System package inventory differs",
    )
    for row in data["system_packages"].values():
        verify_leaf(row, roots["runtime"])
    cpp = data["generated_cpp"]
    require(
        cpp is None
        or (type(cpp) is list and cpp and len({row["path"] for row in cpp}) == len(cpp)),
        "generated C++ inventory differs",
    )
    for row in cpp or []:
        require(
            verify_leaf(row, roots["runtime"]).suffix == ".cpp",
            "generated C++ path is not an actual cpp file",
        )
    for native in provenance["native_by_rank"]:
        require(
            native["native_path"] == data["native"]["path"]
            and native["native_sha256"] == data["native"]["sha256"],
            "executed native origin differs",
        )
        packages = native["system_packages"]
        require(
            len(packages) == 2 and {row["block"] for row in packages} == {"dual", "target"},
            "executed package block inventory differs",
        )
        for row in packages:
            require(
                row["binary_sha256"] == data["system_packages"][row["block"]]["sha256"]
                and type(row["abi_version"]) is int
                and row["abi_version"] == 7,
                "executed System package digest/ABI differs",
            )
    return {
        "metadata": leaf(path, roots["archive"]),
        "evidence": data,
        "generated_cpp_mapping": "not_stored"
        if cpp is None
        else "owner_supplied_actual_files_not_expression_mapping",
        "limits": [
            "Python/SDK origins are execution-owner-attested: fixture records no Python origin or SDK file path",
            "generated C++ semantic mapping is not reconstructed from missing cache files",
            "safe_rebind is a fresh bind, not a successful retry of the impossible outside runtime",
        ],
    }


def assemble(
    *,
    archive_root,
    data_directory,
    source_root,
    installation_root,
    runtime_root,
    sources,
    junit,
    metadata,
):
    roots = {
        key: root_directory(value)
        for key, value in {
            "archive": archive_root,
            "source": source_root,
            "installation": installation_root,
            "runtime": runtime_root,
        }.items()
    }
    require(set(sources) == {"fixture", "example", "snapshot"}, "source inventory differs")
    phases = phase_inventory(data_directory, roots["archive"])
    provenance_path = contained(Path(data_directory) / "provenance.json", roots["archive"])
    provenance = strict_json(provenance_path.read_bytes())
    require(
        provenance.get("schema") == "sol61.m18-native-provenance@1"
        and type(provenance.get("ranks")) is int
        and provenance["ranks"] in (1, 2)
        and type(provenance.get("dimension")) is int
        and provenance["dimension"] == 2,
        "provenance rank/dimension/schema mismatch",
    )
    ranks = provenance["ranks"]
    require(
        provenance.get("constants") == oracle.CONSTANTS
        and len(provenance["native_by_rank"]) == ranks,
        "native constants/rank inventory differs",
    )
    native = provenance["native_by_rank"][0]
    require(
        all(row == native for row in provenance["native_by_rank"])
        and type(native["native_dimension"]) is int
        and native["native_dimension"] == 2
        and type(native["native_capabilities"]["abi_version"]) is int
        and native["native_capabilities"]["abi_version"] == 5,
        "native per-rank identity differs",
    )
    origin = origin_metadata(metadata, roots, provenance)
    checked_sources = {name: leaf(path, roots["source"]) for name, path in sources.items()}
    require(
        provenance["source_sha256"] == {key: row["sha256"] for key, row in checked_sources.items()},
        "executed source digest differs",
    )
    oracle.authored_contract(Path(sources["example"]).read_text())
    oracle.compiled_contract(provenance["program_ir"], provenance["program_ir_hash"])
    require(
        provenance.get("platform")
        and provenance.get("compiled_plan")
        and provenance.get("compiled_components"),
        "compiled platform/plan/component evidence absent",
    )
    require(
        len(junit) == ranks and len(set(map(str, junit))) == ranks, "JUnit rank inventory differs"
    )
    pins = dict(
        schema="sol61.m18-owner-pins@1",
        source_commit=origin["evidence"]["source_commit"],
        native_sha256=native["native_sha256"],
        abi_key=native["abi_key"],
        dimension=2,
        ranks=ranks,
        artifact_identity=provenance["artifact_identity"],
        native_capability_abi=5,
        native_system_package_abi=7,
        provenance=leaf(provenance_path, roots["archive"]),
        sources=checked_sources,
        junit_by_rank=[leaf(path, roots["archive"]) for path in junit],
        phases=phases,
    )
    rows = [
        pins["provenance"],
        *pins["sources"].values(),
        *pins["junit_by_rank"],
        *(row for phase in phases.values() for row in phase.values()),
    ]
    require(len({row["path"] for row in rows}) == 34 + ranks, "pinned files overlap")
    for rank, path in enumerate(junit):
        strict_junit(Path(path).read_bytes(), pins, rank, data_directory)
    # Existing checkpoint/receipt/support reader, without scientific accepted-state predicates.
    for phase in oracle.PHASES:
        oracle.snapshot(Path(data_directory), pins, phase)
    return dict(
        schema="sol61.m18-owner-pending@1",
        status="pending_external_ROOT_approval",
        roots={key: str(value) for key, value in roots.items()},
        data_directory=str(data_directory),
        origin=origin,
        pins=pins,
        proposed_pins_sha256=digest(encoded(pins)),
        native_execution_in_assembler=False,
        scientific_reception=False,
    )


def approved_pins(template_path, approval_path, external_approval_sha):
    require(
        type(external_approval_sha) is str and re.fullmatch("[0-9a-f]{64}", external_approval_sha),
        "external ROOT approval seal required",
    )
    raw, approval_raw = Path(template_path).read_bytes(), Path(approval_path).read_bytes()
    require(digest(approval_raw) == external_approval_sha, "external ROOT approval SHA differs")
    template, approval = strict_json(raw), strict_json(approval_raw)
    require(
        type(approval) is dict
        and set(approval) == {"schema", "status", "approved_by", "pending_sha256", "pins_sha256"}
        and approval["schema"] == "sol61.m18-root-approval@1"
        and approval["status"] == "approved"
        and approval["approved_by"] == "ROOT"
        and approval["pending_sha256"] == digest(raw),
        "ROOT has not approved this exact pending template",
    )
    require(
        template["schema"] == "sol61.m18-owner-pending@1"
        and template["status"] == "pending_external_ROOT_approval"
        and template["scientific_reception"] is False
        and template["native_execution_in_assembler"] is False,
        "pending template contract differs",
    )
    pins = template["pins"]
    require(
        approval["pins_sha256"] == template["proposed_pins_sha256"] == digest(encoded(pins)),
        "approved pin image differs",
    )
    roots = {key: root_directory(value) for key, value in template["roots"].items()}
    require(
        phase_inventory(template["data_directory"], roots["archive"]) == pins["phases"],
        "phase files changed after assembly",
    )
    for row in [pins["provenance"], *pins["junit_by_rank"]]:
        verify_leaf(row, roots["archive"])
    for row in pins["sources"].values():
        verify_leaf(row, roots["source"])
    verify_leaf(template["origin"]["metadata"], roots["archive"])
    provenance = strict_json(Path(pins["provenance"]["path"]).read_bytes())
    require(
        origin_metadata(template["origin"]["metadata"]["path"], roots, provenance)
        == template["origin"],
        "origin files changed after assembly",
    )
    return pins


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(data))
    print(
        json.dumps(
            {
                "path": str(path.resolve()),
                "sha256": digest(path.read_bytes()),
                "scientific_reception": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("assemble")
    for name in (
        "archive-root",
        "data-directory",
        "source-root",
        "installation-root",
        "runtime-root",
        "metadata",
        "output",
    ):
        build.add_argument("--" + name, type=Path, required=True)
    for name in ("fixture", "example", "snapshot"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument(
        "--junit",
        type=Path,
        action="append",
        required=True,
        help="one Serial XML or rank0 then rank1 MPI2 XML",
    )
    approve = sub.add_parser("approve")
    for name in ("pending", "approval", "output"):
        approve.add_argument("--" + name, type=Path, required=True)
    approve.add_argument(
        "--approval-sha256",
        required=True,
        help="SHA externally communicated by ROOT, never self-approved",
    )
    args = parser.parse_args()
    if args.command == "assemble":
        require(
            not args.output.resolve().is_relative_to(args.data_directory.resolve()),
            "pending output must be outside the closed data directory",
        )
        result = assemble(
            archive_root=args.archive_root,
            data_directory=args.data_directory,
            source_root=args.source_root,
            installation_root=args.installation_root,
            runtime_root=args.runtime_root,
            sources={key: getattr(args, key) for key in ("fixture", "example", "snapshot")},
            junit=args.junit,
            metadata=args.metadata,
        )
    else:
        pending = strict_json(args.pending.read_bytes())
        require(
            not args.output.resolve().is_relative_to(Path(pending["data_directory"])),
            "owner pins output must be outside the closed data directory",
        )
        result = approved_pins(args.pending, args.approval, args.approval_sha256)
    write_new(args.output, result)


if __name__ == "__main__":
    main()
