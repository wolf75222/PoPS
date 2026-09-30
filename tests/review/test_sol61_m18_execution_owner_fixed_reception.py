"""Exact d6ec capture reception; opaque protocol files are never native evidence."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TARGET = "d6ec6706895c39f2a0d6ad7f9ae5332df16cab83"
HISTORICAL = "5cbcdf4c5ada9627305d6020e629fb9daccd65f1"


def frozen_module(name, revision, path, *, filename=None):
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", revision + ":" + path])
    module = ModuleType(name)
    module.__file__ = str(filename or ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


history, _ = frozen_module("m18_history_fixture_fixed_review", HISTORICAL,
                          "tests/review/test_sol61_m18_execution_owner_independent.py",
                          filename=Path(__file__))
helper, SOURCE_HASH = frozen_module("m18_corrected_capture_fixed_review", TARGET,
                                  "tests/python/support/m18_entropy_receipts.py")


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    return history.protocol.__wrapped__(tmp_path, monkeypatch)


def save(protocol, world=None):
    return helper.save_entropy_execution_owner(world, protocol.artifact, protocol.directory,
                                              fixture=__file__)


def sidecar(protocol):
    return protocol.directory.parent / "m18-execution-owner-metadata.json"


def select(protocol, kind, path):
    if kind == "native":
        protocol.selector.selected_native_module = lambda **_: SimpleNamespace(__file__=path)
    elif kind == "generated_cpp":
        protocol.artifact.program.generated_sources = [path]
    elif kind in ("dual", "target"):
        next(block for block in protocol.artifact.blocks if block.name == kind).model.so_path = path
    elif kind == "python_package":
        sys.modules["pops"].__file__ = str(path)
    elif kind == "sdk":
        sys.modules["pops.codegen.toolchain"].pops_include = lambda: path.parent


@pytest.mark.parametrize("kind", ("native", "generated_cpp", "dual", "target", "python_package", "sdk"))
def test_exact_same_symbolic_attack_81249_accepted_d6ec_refuses_before_output(protocol, kind):
    target = {"native": protocol.native, "generated_cpp": protocol.cpp,
              "python_package": protocol.package, "sdk": protocol.sdk,
              **protocol.binaries}[kind]
    alias = target.with_name("symbolic" + target.suffix)
    alias.symlink_to(target)
    if kind == "sdk":
        directory_alias = target.parent.with_name("symbolic-include")
        directory_alias.symlink_to(target.parent, target_is_directory=True)
        alias = directory_alias / target.name
    select(protocol, kind, alias)
    old = history.helper._execution_owner_record(protocol.artifact, __file__)
    assert old[kind if kind not in ("generated_cpp", "dual", "target") else
               "generated_cpp" if kind == "generated_cpp" else "system_packages"]
    with pytest.raises(AssertionError, match="M18 execution origin path aliases are forbidden"):
        save(protocol)
    assert not sidecar(protocol).exists() and list(protocol.directory.iterdir()) == []


@pytest.mark.parametrize("alias_kind", ("parent", "dangling", "dotdot"))
def test_cpp_alias_is_rejected_before_missing_file_fallback(protocol, alias_kind):
    if alias_kind == "parent":
        parent = protocol.cpp.parent.with_name("parent-alias")
        parent.symlink_to(protocol.cpp.parent, target_is_directory=True)
        alias = parent / protocol.cpp.name
    elif alias_kind == "dangling":
        alias = protocol.cpp.with_name("dangling.cpp")
        alias.symlink_to(protocol.cpp.with_name("absent.cpp"))
    else:
        alias = protocol.cpp.parent / ".." / protocol.cpp.parent.name / protocol.cpp.name
    protocol.artifact.program.generated_sources = [alias]
    with pytest.raises(AssertionError, match="M18 execution origin path aliases are forbidden"):
        save(protocol)
    assert not sidecar(protocol).exists()


@pytest.mark.parametrize("exists", (True, False))
def test_original_single_program_duplicate_refuses_before_union_or_absence(protocol, exists):
    if not exists:
        protocol.cpp.unlink()
    protocol.artifact.program.generated_sources = [protocol.cpp, protocol.cpp]
    old = history.helper._execution_owner_record(protocol.artifact, __file__)
    assert old["generated_cpp"] == ([history.helper._owner_file(protocol.cpp)] if exists else None)
    with pytest.raises(AssertionError, match="M18 Program generated source inventory contains duplicates"):
        save(protocol)
    assert not sidecar(protocol).exists()


@pytest.mark.parametrize("kind", ("native", "python_package", "sdk", "dual", "target"))
def test_each_non_cpp_byte_drift_refuses_in_revalidation_before_output(protocol, monkeypatch, kind):
    original = helper._owner_consensus
    path = {"native": protocol.native, "python_package": protocol.package,
            "sdk": protocol.sdk, **protocol.binaries}[kind]

    def changed(records):
        result = original(records)
        path.write_bytes(b"CHANGED OPAQUE ORIGIN AFTER CAPTURE")
        return result

    monkeypatch.setattr(helper, "_owner_consensus", changed)
    with pytest.raises(AssertionError, match="M18 execution origins changed before recording"):
        save(protocol)
    assert not sidecar(protocol).exists()


@pytest.mark.parametrize("kind", ("provider_path", "source_head", "local_cpp_metadata"))
def test_origin_metadata_drift_is_not_reauthenticated_as_a_new_authority(protocol, monkeypatch, kind):
    original = helper._owner_consensus

    def changed(records):
        result = original(records)
        if kind == "provider_path":
            different = protocol.native.with_name("other-native.so")
            different.write_bytes(protocol.native.read_bytes())
            select(protocol, "native", different)
        elif kind == "source_head":
            monkeypatch.setattr(helper, "_fixture_commit", lambda _: "0" * 40)
        else:
            protocol.artifact.program.generated_sources = []
        return result

    monkeypatch.setattr(helper, "_owner_consensus", changed)
    with pytest.raises(AssertionError, match="M18 execution origins changed before recording"):
        save(protocol)
    assert not sidecar(protocol).exists()


@pytest.mark.parametrize("rank", (0, 1, 2))
def test_rank_order_rehash_follows_path_vote_precedes_final_record_vote_and_write(protocol, monkeypatch, rank):
    events = []
    initial = helper._execution_owner_record

    def recorded(*args):
        result = initial(*args)
        events.append("capture")
        return result

    def gather(world, value):
        events.append("record_vote" if isinstance(value, dict) else
                      "path_vote" if isinstance(value, str) else "error_vote")
        return (value,) * world.size

    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        if path == sidecar(protocol):
            assert args[0] == "x"
            events.append("write")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(helper, "_execution_owner_record", recorded)
    monkeypatch.setattr(Path, "open", tracked_open)
    protocol.collectives.allgather_value = gather
    save(protocol, SimpleNamespace(rank=rank, size=3))
    important = [event for event in events if event != "error_vote"]
    assert important == ["capture", "record_vote", "path_vote", "capture", "record_vote"] + (
        ["write"] if rank == 0 else [])
    assert events[-1] == "error_vote"  # non-contributors also enter post-publication convergence
    assert sidecar(protocol).exists() == (rank == 0)


def test_shared_cpp_across_two_programs_and_cache_peer_are_admissible(protocol):
    program = protocol.artifact.program
    protocol.artifact.layout_programs = (SimpleNamespace(program=program), SimpleNamespace(program=program))
    record = helper._execution_owner_record(protocol.artifact, __file__)
    assert record["generated_cpp"] == [helper._owner_file(protocol.cpp)]
    protocol.artifact.layout_programs = ()
    protocol.artifact.program.generated_sources = []
    peer = helper._execution_owner_record(protocol.artifact, __file__)

    def gather(world, value):
        if isinstance(value, dict):
            return record, peer if value["generated_cpp"] is None else record
        return (value,) * world.size

    protocol.collectives.allgather_value = gather
    save(protocol, SimpleNamespace(rank=1, size=2))
    assert not sidecar(protocol).exists()


def test_real_missing_metadata_stays_null_and_original_schema_is_unchanged(protocol):
    protocol.artifact.program.generated_sources = []
    record = helper._execution_owner_record(protocol.artifact, __file__)
    assert protocol.cpp.exists() and record["generated_cpp"] is None
    assert record == history.helper._execution_owner_record(protocol.artifact, __file__)
    path = save(protocol)
    assert json.loads(path.read_bytes()) == record
    assert not {"status", "approved_by", "native_result"}.intersection(record)
    assert SOURCE_HASH == hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(ROOT), "show", TARGET + ":tests/python/support/m18_entropy_receipts.py"])).hexdigest()


def test_cpp_drift_retains_elected_diagnostic_before_write(protocol, monkeypatch):
    original = helper._owner_consensus

    def changed(records):
        result = original(records)
        protocol.cpp.write_bytes(b"CHANGED OPAQUE CPP")
        return result

    monkeypatch.setattr(helper, "_owner_consensus", changed)
    with pytest.raises(AssertionError, match="M18 elected generated source changed before recording"):
        save(protocol)
    assert not sidecar(protocol).exists()


@pytest.mark.parametrize("rank", (0, 1, 2))
def test_non_cpp_change_after_path_vote_converges_before_final_record_vote(protocol, rank):
    events = []

    def gather(world, value):
        if isinstance(value, dict):
            events.append("record_vote")
        elif isinstance(value, str):
            events.append("path_vote")
            protocol.native.write_bytes(b"OPAQUE NATIVE CHANGED AFTER DESTINATION VOTE")
        else:
            events.append("failure_vote" if value is not None else "error_vote")
        return (value,) * world.size

    protocol.collectives.allgather_value = gather
    with pytest.raises(AssertionError, match="M18 execution origins changed before recording"):
        save(protocol, SimpleNamespace(rank=rank, size=3))
    assert events.count("record_vote") == 1 and events.count("path_vote") == 1
    assert events[-1] == "failure_vote"
    assert not sidecar(protocol).exists()
