"""Protocol-only controls: synthetic XML/bytes never claim native or scientific success."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests/review/sol61_m18_owner_assemble.py"
spec = importlib.util.spec_from_file_location("m18_owner_assembler_unit", SCRIPT)
assembler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = assembler
spec.loader.exec_module(assembler)


def junit(size=1, rank=0, directory="/ACTUAL-RECEIPT-REALM"):
    # Deliberately synthetic protocol XML; these identities are not native receipts.
    pins = dict(
        ranks=size,
        native_sha256="a" * 64,
        abi_key="PROTOCOL-ONLY",
        artifact_identity="PROTOCOL-ONLY",
    )
    suite = ET.Element("testsuites")
    ET.SubElement(suite, "testsuite", tests="1", failures="0", errors="0", skipped="0")
    case = ET.SubElement(suite.find("testsuite"), "testcase", name=assembler.oracle.TEST_NAME)
    properties = ET.SubElement(case, "properties")
    for name, value in dict(
        dim=2,
        rank=rank,
        mpi_ranks=size,
        native_sha256=pins["native_sha256"],
        native_abi_key=pins["abi_key"],
        native_capability_abi=5,
        artifact_identity=pins["artifact_identity"],
        saved_receipts=directory,
        outside_cone_failure_0="PROTOCOL-ONLY",
        outside_cone_failure_1="PROTOCOL-ONLY",
    ).items():
        ET.SubElement(properties, "property", name=name, value=str(value))
    return suite, pins


@pytest.mark.parametrize("size,rank", ((1, 0), (2, 0), (2, 1)))
def test_strict_junit_single_witness_each_rank_only_protocol(size, rank):
    suite, pins = junit(size, rank)
    result = assembler.strict_junit(ET.tostring(suite), pins, rank, Path("/ACTUAL-RECEIPT-REALM"))
    assert result["rank"] == str(rank)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra",
        "foreign",
        "failure",
        "error",
        "skipped",
        "aggregate",
        "duplicate",
        "rank",
        "realm",
        "refusals",
    ),
)
def test_junit_closed_inventory_status_rank_properties_and_realm(mutation):
    suite, pins = junit(2, 0)
    case = suite.find(".//testcase")
    props = case.find("properties")
    if mutation == "extra":
        ET.SubElement(suite, "testcase", name="foreign")
    elif mutation == "foreign":
        case.set("name", "foreign")
    elif mutation in ("failure", "error", "skipped"):
        ET.SubElement(case, mutation)
    elif mutation == "aggregate":
        suite.set("failures", "1")
    elif mutation == "duplicate":
        props.append(copy.deepcopy(props.find("property")))
    else:
        name = {"rank": "rank", "realm": "saved_receipts", "refusals": "outside_cone_failure_0"}[
            mutation
        ]
        node = next(row for row in props if row.get("name") == name)
        if mutation == "refusals":
            props.remove(node)
        else:
            node.set("value", "foreign")
    with pytest.raises(ValueError):
        assembler.strict_junit(ET.tostring(suite), pins, 0, Path("/ACTUAL-RECEIPT-REALM"))


def inventory(folder):
    folder.mkdir()
    (folder / "provenance.json").write_text("{}")
    for phase in assembler.oracle.PHASES:
        # Opaque protocol bytes only, never NPZ arrays or invented accepted native state.
        checkpoint = folder / (phase + ".npz")
        checkpoint.write_bytes(b"NOT-NATIVE-NPZ")
        (folder / (phase + "-state.npz")).write_bytes(b"NOT-NATIVE-NPZ")
        (folder / (phase + "-receipt.json")).write_bytes(
            assembler.encoded(
                dict(
                    phase=phase,
                    checkpoint=checkpoint.name,
                    checkpoint_sha256=assembler.digest(checkpoint.read_bytes()),
                )
            )
        )


def test_ten_phase_filename_inventory_is_pending_only(tmp_path):
    folder = tmp_path / "data"
    inventory(folder)
    result = assembler.phase_inventory(folder, tmp_path)
    assert (
        set(result) == set(assembler.oracle.PHASES)
        and sum(len(row) for row in result.values()) == 30
    )
    # These protocol-only receipts lack actual artifact provenance and cannot reach science.
    with pytest.raises(KeyError, match="artifact_identity"):
        assembler.oracle.snapshot(
            folder, dict(phases=result, artifact_identity="NOT-NATIVE", ranks=1), "initial"
        )


@pytest.mark.parametrize(
    "mutation", ("extra", "missing", "foreign", "escape", "digest", "reused", "symlink")
)
def test_phase_inventory_refuses_unclosed_foreign_aliased_or_changed_inputs(tmp_path, mutation):
    folder = tmp_path / "data"
    inventory(folder)
    receipt = folder / "accepted-receipt.json"
    data = json.loads(receipt.read_text())
    if mutation == "extra":
        (folder / "unowned-state.npz").write_bytes(b"NOT-NATIVE")
    elif mutation == "missing":
        (folder / "initial-state.npz").unlink()
    elif mutation == "foreign":
        data["phase"] = "safe_rebind_accepted"
    elif mutation == "escape":
        data["checkpoint"] = "../foreign.npz"
    elif mutation == "digest":
        (folder / "accepted.npz").write_bytes(b"CHANGED")
    elif mutation == "reused":
        data["checkpoint"] = "initial.npz"
    elif mutation == "symlink":
        (folder / "accepted.npz").unlink()
        (folder / "accepted.npz").symlink_to(folder / "initial.npz")
    receipt.write_bytes(assembler.encoded(data))
    with pytest.raises(ValueError):
        assembler.phase_inventory(folder, tmp_path)


@pytest.mark.parametrize("kind", ("escape", "relative", "alias", "parent_symlink", "changed"))
def test_strict_roots_and_digest_refuse_source_or_package_escapes(tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    p = root / "fixture.py"
    p.write_text("SOURCE-ONLY")
    row = assembler.leaf(p, root)
    if kind == "changed":
        p.write_text("CHANGED")
        with pytest.raises(ValueError):
            assembler.verify_leaf(row, root)
        return
    if kind == "escape":
        q = tmp_path / "foreign"
        q.write_text("FOREIGN")
    elif kind == "relative":
        q = Path("fixture.py")
    elif kind == "alias":
        q = root / ".." / "root" / "fixture.py"
    else:
        (root / "alias").symlink_to(root, target_is_directory=True)
        q = root / "alias" / "fixture.py"
    with pytest.raises(ValueError):
        assembler.leaf(q, root)


@pytest.mark.parametrize("seal", (None, "", "f" * 63, "g" * 64))
def test_approval_requires_external_seal_before_reading_files(tmp_path, seal):
    with pytest.raises(ValueError):
        assembler.approved_pins(tmp_path / "missing", tmp_path / "missing", seal)


@pytest.mark.parametrize("change", ("pending", "owner", "status", "pins", "seal"))
def test_root_approval_is_bound_to_exact_pending_bytes_and_pin_image(tmp_path, change):
    template = dict(
        schema="sol61.m18-owner-pending@1",
        status="pending_external_ROOT_approval",
        scientific_reception=False,
        native_execution_in_assembler=False,
        pins={"schema": "NOT-A-NATIVE-OWNER"},
        proposed_pins_sha256=assembler.digest(assembler.encoded({"schema": "NOT-A-NATIVE-OWNER"})),
    )
    pending = tmp_path / "pending.json"
    pending.write_bytes(assembler.encoded(template))
    approval = dict(
        schema="sol61.m18-root-approval@1",
        status="approved",
        approved_by="ROOT",
        pending_sha256=assembler.digest(pending.read_bytes()),
        pins_sha256=template["proposed_pins_sha256"],
    )
    if change == "pending":
        pending.write_bytes(pending.read_bytes() + b" ")
    elif change == "owner":
        approval["approved_by"] = "SELF"
    elif change == "status":
        approval["status"] = "pending"
    elif change == "pins":
        approval["pins_sha256"] = "f" * 64
    path = tmp_path / "approval.json"
    path.write_bytes(assembler.encoded(approval))
    seal = "f" * 64 if change == "seal" else assembler.digest(path.read_bytes())
    with pytest.raises(ValueError):
        assembler.approved_pins(pending, path, seal)


def test_cli_never_imports_pops_or_creates_scientific_files(tmp_path):
    code = """import importlib.abc,runpy,sys
class NoPoPS(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=="pops" or name.startswith("pops."):raise AssertionError("PoPS forbidden")
sys.meta_path.insert(0,NoPoPS());sys.argv=[sys.argv[1],"--help"];runpy.run_path(sys.argv[0],run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "assemble" in result.stdout and "approve" in result.stdout
    assert list(tmp_path.iterdir()) == []


def origins(tmp_path):
    roots = {name: tmp_path / name for name in ("archive", "installation", "runtime", "source")}
    for path in roots.values():
        path.mkdir()
    files = {}
    for kind in ("python_package", "sdk", "native"):
        path = roots["installation"] / (kind + ".PROTOCOL-ONLY")
        path.write_bytes(b"NOT-NATIVE-" + kind.encode())
        files[kind] = assembler.leaf(path, roots["installation"])
    files["system_packages"] = {}
    for block in ("dual", "target"):
        path = roots["runtime"] / (block + ".PROTOCOL-ONLY")
        path.write_bytes(b"NOT-A-SYSTEM-PACKAGE-" + block.encode())
        files["system_packages"][block] = assembler.leaf(path, roots["runtime"])
    metadata = dict(
        schema="sol61.m18-execution-owner-metadata@1",
        source_commit="a" * 40,
        generated_cpp=None,
        **files,
    )
    path = roots["archive"] / "metadata.json"
    path.write_bytes(assembler.encoded(metadata))
    native = dict(
        native_path=files["native"]["path"],
        native_sha256=files["native"]["sha256"],
        system_packages=[
            dict(block=block, abi_version=7, binary_sha256=row["sha256"])
            for block, row in files["system_packages"].items()
        ],
    )
    return roots, path, metadata, dict(native_by_rank=[native, native])


def test_absent_generated_cpp_is_explicitly_unmapped_protocol_only(tmp_path):
    roots, path, _, provenance = origins(tmp_path)
    result = assembler.origin_metadata(path, roots, provenance)
    assert result["generated_cpp_mapping"] == "not_stored"
    assert result["evidence"]["generated_cpp"] is None
    assert result["metadata"]["sha256"] == assembler.digest(path.read_bytes())
    assert "no Python origin" in result["limits"][0]


@pytest.mark.parametrize(
    "mutation",
    ("source_commit", "native", "package", "missing_package", "SDK_changed", "generated_escape"),
)
def test_owner_origin_metadata_refuses_foreign_sources_packages_or_cpp(tmp_path, mutation):
    roots, path, data, provenance = origins(tmp_path)
    if mutation == "source_commit":
        data["source_commit"] = "not-a-commit"
    elif mutation == "native":
        provenance["native_by_rank"][1] = copy.deepcopy(provenance["native_by_rank"][0])
        provenance["native_by_rank"][1]["native_path"] = str(roots["installation"] / "foreign")
    elif mutation == "package":
        provenance["native_by_rank"][0]["system_packages"][0]["binary_sha256"] = "f" * 64
    elif mutation == "missing_package":
        del data["system_packages"]["target"]
    elif mutation == "SDK_changed":
        Path(data["sdk"]["path"]).write_bytes(b"CHANGED")
    else:
        escaped = tmp_path / "outside.cpp"
        escaped.write_text("NOT-NATIVE-CXX")
        data["generated_cpp"] = [
            dict(path=str(escaped), sha256=assembler.digest(escaped.read_bytes()))
        ]
    path.write_bytes(assembler.encoded(data))
    with pytest.raises(ValueError):
        assembler.origin_metadata(path, roots, provenance)
