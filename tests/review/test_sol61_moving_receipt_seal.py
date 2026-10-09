"""Negative inventory/authority probes; synthetic XML is not physical evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("sol61_moving_receipt_seal", HERE / "sol61_moving_receipt_seal.py")
seal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seal)


def xml_file(tmp_path, *, skip=False, rank="0", duplicate=False):
    suite = ET.Element("testsuite", tests="8", failures="0", errors="0", skipped="0")
    for name in sorted(seal.expected_nodes()):
        node = ET.SubElement(suite, "testcase", classname=seal.CLASS, name=name)
        props = ET.SubElement(node, "properties")
        for key, value in (("rank", rank), ("size", "1"), ("dimension", "1")):
            ET.SubElement(props, "property", name=key, value=value)
        if duplicate:
            ET.SubElement(props, "property", name="rank", value="0")
        if skip:
            ET.SubElement(node, "skipped")
    path = tmp_path / "software-only.xml"
    ET.ElementTree(suite).write(path)
    return path


@pytest.mark.parametrize("mode,diagnostic", [("skip", "did not pass"), ("rank", "rank/size"),
                                            ("duplicate", "duplicate JUnit property")])
def test_rejects_junit_before_unavailable_physical_receipts(tmp_path, mode, diagnostic):
    path = xml_file(tmp_path, skip=mode == "skip", rank="1" if mode == "rank" else "0",
                    duplicate=mode == "duplicate")
    with pytest.raises(ValueError, match=diagnostic):
        seal.junit_cases(tmp_path, path, 0, 1)


def test_rejects_incomplete_node_inventory(tmp_path):
    path = tmp_path / "incomplete.xml"
    path.write_text('<testsuite><testcase name="one"/></testsuite>')
    with pytest.raises(ValueError, match="exact eight"):
        seal.junit_cases(tmp_path, path, 0, 1)


def test_absolute_path_and_symlink_escape_refuse(tmp_path):
    launch = tmp_path / "launch"
    launch.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"test-only")
    (launch / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        seal.donor(launch, launch / "escape")
    with pytest.raises(ValueError, match="explicit and absolute"):
        seal.donor(launch, Path("escape"))


@pytest.mark.parametrize("rank,size,authority", [(0, 1, "pending_external_owner_approval"),
                                               (True, 1, "external_owner"), (0, True, "external_owner"),
                                               (1, 1, "external_owner")])
def test_external_owner_and_exact_integer_authority_required(tmp_path, rank, size, authority):
    owner = dict(schema=seal.SCHEMA, authority=authority, rank=rank, size=size, dimension=1)
    with pytest.raises(ValueError, match="external owner/rank/dimension"):
        seal.assemble(tmp_path, 0, owner)


def test_narrow_binary_cbor_is_type_aware_and_identity_domain_is_closed():
    assert seal.cbor_bytes({"x": bytes(range(32))}) == b"\xa1\x61x\x58\x20" + bytes(range(32))
    with pytest.raises(ValueError, match="32 bytes"):
        seal.cbor_bytes(bytes(31))
    with pytest.raises(ValueError, match="domain/version"):
        seal.identity_data("pops.binary.v2:sha256:" + "0" * 64, "binary")


def test_helper_and_oracle_never_import_pops_in_an_isolated_process(tmp_path):
    script = (HERE / "sol61_moving_receipt_seal.py").resolve()
    code = '''import importlib.abc, importlib.util, pathlib, sys
class RefusePoPS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pops" or fullname.startswith("pops."):
            raise AssertionError("PoPS/native import forbidden: " + fullname)
sys.meta_path.insert(0, RefusePoPS())
helper = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(helper.parent))
spec = importlib.util.spec_from_file_location("isolated_moving_seal", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.expected_nodes()) == 8
assert module.oracle.contract()["consumes_no_pops_package"] is True
assert not any(name == "pops" or name.startswith("pops.") for name in sys.modules)
print("helper+oracle isolated without PoPS/native imports")
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "helper+oracle isolated without PoPS/native imports"
