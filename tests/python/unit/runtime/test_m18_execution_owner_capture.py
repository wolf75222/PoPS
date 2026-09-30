"""Execution-owner capture source protocol; dummy files are never native evidence."""
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pops
from pops import _native_collectives
from pops import _native_selector
from pops.codegen import toolchain
from tests.python.support import m18_entropy_receipts as helper


def origins(tmp_path, monkeypatch, *, sources=True):
    installation = tmp_path / "installation"
    installation.mkdir()
    package = installation / "__init__.py"
    package.write_bytes(b"test Python entry")
    native = installation / "test-native.so"
    native.write_bytes(b"dummy protocol file, no native execution")
    sdk = installation / "include"
    sdk.mkdir()
    (sdk / "pops_headers.manifest").write_bytes(b"actual test manifest file")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    blocks = []
    for name in ("dual", "target"):
        binary = runtime / (name + ".so")
        binary.write_bytes(("dummy test " + name).encode())
        blocks.append(SimpleNamespace(name=name, model=SimpleNamespace(so_path=binary)))
    cpp = runtime / "recorded-program.cpp"
    cpp.write_bytes(b"// actual retained test file\n")
    # This plausible cache sibling must never be discovered as generated source.
    (runtime / "dual.cpp").write_bytes(b"not recorded in compiler metadata")
    program = SimpleNamespace(generated_sources=[str(cpp)] if sources else [])
    artifact = SimpleNamespace(blocks=blocks, program=program, layout_programs=())
    monkeypatch.setattr(sys, "prefix", str(installation))
    monkeypatch.setattr(pops, "__file__", str(package))
    monkeypatch.setattr(toolchain, "pops_include", lambda: str(sdk))
    monkeypatch.setattr(_native_selector, "selected_native_module", lambda **_: SimpleNamespace(__file__=native))
    return artifact, helper._execution_owner_record(artifact, __file__), cpp


def test_real_file_paths_and_commit_are_captured_without_cache_glob(tmp_path, monkeypatch):
    artifact, record, cpp = origins(tmp_path, monkeypatch)
    assert re.fullmatch("[0-9a-f]{40}", record["source_commit"])
    assert set(record) == {"schema", "source_commit", "python_package", "sdk", "native", "system_packages", "generated_cpp"}
    assert record["generated_cpp"] == [helper._owner_file(cpp)]
    assert {name: row["path"] for name, row in record["system_packages"].items()} == {
        block.name: str(Path(block.model.so_path).resolve()) for block in artifact.blocks}
    for name in ("python_package", "sdk", "native"):
        assert record[name] == helper._owner_file(record[name]["path"])


@pytest.mark.parametrize("missing", ("metadata", "file"))
def test_unavailable_recorded_source_is_null_even_when_cache_sibling_exists(tmp_path, monkeypatch, missing):
    artifact, _, cpp = origins(tmp_path, monkeypatch, sources=missing != "metadata")
    if missing == "file":
        cpp.unlink()
    assert helper._execution_owner_record(artifact, __file__)["generated_cpp"] is None


@pytest.mark.parametrize("defect", ("duplicate", "missing", "foreign"))
def test_system_inventory_refuses_before_recording(tmp_path, monkeypatch, defect):
    artifact, _, _ = origins(tmp_path, monkeypatch)
    if defect == "duplicate":
        artifact.blocks.append(artifact.blocks[0])
    elif defect == "missing":
        artifact.blocks.pop()
    else:
        artifact.blocks[0].name = "foreign"
    with pytest.raises(ValueError, match="(?:inventory|dual and target)"):
        helper._execution_owner_record(artifact, __file__)


def test_installed_origins_refuse_source_package_substitution(tmp_path, monkeypatch):
    artifact, _, _ = origins(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign.py"
    foreign.write_text("not installed")
    monkeypatch.setattr(pops, "__file__", str(foreign))
    with pytest.raises(ValueError, match="actual installed"):
        helper._execution_owner_record(artifact, __file__)


def test_cache_hit_peer_authenticates_direct_elected_source_paths(tmp_path, monkeypatch):
    _, record, _ = origins(tmp_path, monkeypatch)
    peer = copy.deepcopy(record)
    peer["generated_cpp"] = None
    assert helper._owner_consensus((record, peer)) == record
    assert helper._owner_consensus((peer, peer))["generated_cpp"] is None
    different = copy.deepcopy(record)
    different["generated_cpp"] = [helper._owner_file(tmp_path / "runtime" / "dual.cpp")]
    with pytest.raises(ValueError, match="generated source inventories differ"):
        helper._owner_consensus((record, different))


@pytest.mark.parametrize("field", ("source_commit", "python_package", "sdk", "native", "system_packages"))
def test_foreign_rank_origin_refuses_no_publication(tmp_path, monkeypatch, field):
    _, record, _ = origins(tmp_path, monkeypatch)
    peer = copy.deepcopy(record)
    peer[field] = "different execution origin"
    with pytest.raises(ValueError, match="origins differ"):
        helper._owner_consensus((record, peer))


def test_sidecar_outside_closed_phase_inventory_and_no_overwrite(tmp_path, monkeypatch):
    artifact, record, _ = origins(tmp_path, monkeypatch)
    directory = tmp_path / "phase-dir"
    directory.mkdir()
    (directory / "provenance.json").write_bytes(b"original strict provenance remains unchanged")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    path = helper.save_entropy_execution_owner(None, artifact, directory, fixture=__file__)
    assert path.parent == directory.parent and path.parent != directory
    assert json.loads(path.read_bytes()) == record
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(AssertionError, match="FileExistsError"):
        helper.save_entropy_execution_owner(None, artifact, directory, fixture=__file__)


def test_three_rank_parity_and_empty_peer_reopens_root_source(tmp_path, monkeypatch):
    artifact, record, _ = origins(tmp_path, monkeypatch)
    directory = tmp_path / "phases"
    directory.mkdir()
    peer = copy.deepcopy(record)
    peer["generated_cpp"] = None
    seen = []
    def gather(_world, value):
        seen.append(value)
        if isinstance(value, dict) and value.get("schema") == record["schema"]:
            return (record, peer, peer) if value["generated_cpp"] is None else (value,) * 3
        return (value,) * 3
    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    root = SimpleNamespace(rank=0, size=3)
    path = helper.save_entropy_execution_owner(root, artifact, directory, fixture=__file__)
    artifact.program.generated_sources = []
    root.rank = 2
    assert helper.save_entropy_execution_owner(root, artifact, directory, fixture=__file__) == path
    assert json.loads(path.read_bytes()) == record
    assert record["generated_cpp"][0] == helper._owner_file(record["generated_cpp"][0]["path"])
    assert str(path) in seen


def test_peer_mutated_elected_source_refuses_before_sidecar_write(tmp_path, monkeypatch):
    artifact, record, cpp = origins(tmp_path, monkeypatch)
    directory = tmp_path / "phases"
    directory.mkdir()
    artifact.program.generated_sources = []
    def gather(_world, value):
        if isinstance(value, dict):
            cpp.write_bytes(b"changed after compiler-owner recording")
            return (record, value)
        return (value, value)
    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    with pytest.raises(AssertionError, match="elected generated source changed"):
        helper.save_entropy_execution_owner(SimpleNamespace(rank=1, size=2), artifact, directory, fixture=__file__)
    assert not (tmp_path / "m18-execution-owner-metadata.json").exists()
