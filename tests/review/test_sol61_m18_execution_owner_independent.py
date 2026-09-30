"""Independent protocol counterexamples; opaque bytes never constitute native evidence."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
FROZEN = "81249ef170bba6e8176b3ff54b42e58ae275e24d"


def load(name, path, *, frozen=False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if frozen:
        source = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                          FROZEN + ":" + str(path.relative_to(ROOT))])
        exec(compile(source, str(path), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


helper = load("m18_capture_independent", ROOT / "tests/python/support/m18_entropy_receipts.py", frozen=True)
assembler = load("m18_assembler_independent", ROOT / "tests/review/sol61_m18_owner_assemble.py")


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    roots = {name: tmp_path / name for name in ("installation", "runtime", "archive")}
    for path in roots.values():
        path.mkdir()
    package = roots["installation"] / "entry.py"
    native = roots["installation"] / "native.so"
    sdk_dir = roots["installation"] / "include"
    sdk_dir.mkdir()
    sdk = sdk_dir / "pops_headers.manifest"
    binaries = {name: roots["runtime"] / (name + ".so") for name in ("dual", "target")}
    cpp = roots["runtime"] / "explicit.cpp"
    for path in (package, native, sdk, *binaries.values(), cpp):
        path.write_bytes(b"OPAQUE PROTOCOL ONLY, NOT A NATIVE RESULT: " + path.name.encode())
    pops = ModuleType("pops")
    pops.__file__ = str(package)
    selector = ModuleType("pops._native_selector")
    selector.selected_native_module = lambda **_: SimpleNamespace(__file__=native)
    toolchain = ModuleType("pops.codegen.toolchain")
    toolchain.pops_include = lambda: sdk_dir
    collectives = ModuleType("pops._native_collectives")
    modules = {"pops": pops, "pops._native_selector": selector,
               "pops.codegen": ModuleType("pops.codegen"),
               "pops.codegen.toolchain": toolchain, "pops._native_collectives": collectives}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "prefix", str(roots["installation"]))
    artifact = SimpleNamespace(
        blocks=[SimpleNamespace(name=name, model=SimpleNamespace(so_path=path))
                for name, path in binaries.items()],
        layout_programs=(), program=SimpleNamespace(generated_sources=[cpp]))
    directory = roots["archive"] / "phases"
    directory.mkdir()
    return SimpleNamespace(roots=roots, package=package, native=native, sdk=sdk,
                           binaries=binaries, cpp=cpp, artifact=artifact,
                           collectives=collectives, selector=selector, directory=directory)


def capture(protocol):
    return helper._execution_owner_record(protocol.artifact, __file__)


def provenance(record):
    return {"native_by_rank": [{"native_path": record["native"]["path"],
                               "native_sha256": record["native"]["sha256"],
                               "system_packages": [dict(block=name, abi_version=7,
                                                        binary_sha256=row["sha256"])
                                                   for name, row in record["system_packages"].items()]}]}


def assemble_origin(protocol, record):
    path = protocol.roots["archive"] / "opaque-sidecar.json"
    path.write_text(json.dumps(record))
    return assembler.origin_metadata(path, protocol.roots, provenance(record))


def test_actual_observed_fixture_head_and_explicit_origins_are_not_build_authority(protocol):
    record = capture(protocol)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert record["source_commit"] == head
    assert set(record) == {"schema", "source_commit", "python_package", "sdk", "native",
                           "system_packages", "generated_cpp"}
    assert record["generated_cpp"] == [helper._owner_file(protocol.cpp)]
    assert record["native"]["sha256"] == hashlib.sha256(protocol.native.read_bytes()).hexdigest()
    result = assemble_origin(protocol, record)
    assert "not_expression_mapping" in result["generated_cpp_mapping"]
    assert "status" not in record and "approved_by" not in record


@pytest.mark.parametrize("rank", (0, 1, 2))
def test_all_ranks_finish_identical_boundaries_only_root_opens_output(protocol, monkeypatch, rank):
    events = []
    destination = protocol.directory.parent / "m18-execution-owner-metadata.json"

    def gather(world, value):
        events.append(("gather", value))
        return (value,) * world.size

    protocol.collectives.allgather_value = gather
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        if path == destination:
            events.append(("output", args[0]))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    result = helper.save_entropy_execution_owner(SimpleNamespace(rank=rank, size=3),
                                                protocol.artifact, protocol.directory, fixture=__file__)
    assert result == destination
    assert sum(kind == "gather" for kind, _ in events) == 11
    outputs = [i for i, (kind, _) in enumerate(events) if kind == "output"]
    assert outputs == ([10] if rank == 0 else [])
    assert destination.exists() == (rank == 0)
    assert list(protocol.directory.iterdir()) == []


@pytest.mark.parametrize("phase", ("initial", "confirmation"))
def test_origin_divergence_refuses_before_exclusive_write(protocol, phase):
    count = 0

    def gather(world, value):
        nonlocal count
        if isinstance(value, dict):
            count += 1
            other = copy.deepcopy(value)
            if count == (1 if phase == "initial" else 2):
                other["native"]["sha256"] = "0" * 64
            return value, other
        return (value,) * world.size

    protocol.collectives.allgather_value = gather
    with pytest.raises(AssertionError, match="(?:origins differ|publication differs)"):
        helper.save_entropy_execution_owner(SimpleNamespace(rank=0, size=2),
                                            protocol.artifact, protocol.directory, fixture=__file__)
    assert not (protocol.directory.parent / "m18-execution-owner-metadata.json").exists()


def test_changed_elected_cpp_refuses_before_write(protocol, monkeypatch):
    original = helper._owner_consensus

    def changed(records):
        result = original(records)
        protocol.cpp.write_bytes(b"CHANGED OPAQUE CPP")
        return result

    monkeypatch.setattr(helper, "_owner_consensus", changed)
    with pytest.raises(AssertionError, match="elected generated source changed"):
        helper.save_entropy_execution_owner(None, protocol.artifact, protocol.directory, fixture=__file__)
    assert not (protocol.directory.parent / "m18-execution-owner-metadata.json").exists()


@pytest.mark.parametrize("kind", ("native", "dual"))
def test_counterexample_non_cpp_changes_publish_stale_metadata_but_assembler_refuses(protocol, monkeypatch, kind):
    # A demonstrated capture gap, not a claim of a received/accepted native result.
    original = helper._owner_consensus

    def changed(records):
        result = original(records)
        path = protocol.native if kind == "native" else protocol.binaries["dual"]
        path.write_bytes(b"CHANGED OPAQUE ORIGIN AFTER INITIAL CAPTURE")
        return result

    monkeypatch.setattr(helper, "_owner_consensus", changed)
    path = helper.save_entropy_execution_owner(None, protocol.artifact, protocol.directory, fixture=__file__)
    record = json.loads(path.read_bytes())
    with pytest.raises(ValueError, match="(?:digest|changed|SHA|hash)"):
        assembler.origin_metadata(path, protocol.roots, provenance(record))


@pytest.mark.parametrize("kind", ("native", "generated_cpp"))
def test_counterexample_symbolic_origin_is_erased_and_later_assembler_cannot_refuse(protocol, kind):
    target = protocol.native if kind == "native" else protocol.cpp
    alias = target.with_name("alias" + target.suffix)
    alias.symlink_to(target)
    if kind == "native":
        protocol.selector.selected_native_module = lambda **_: SimpleNamespace(__file__=alias)
    else:
        protocol.artifact.program.generated_sources = [alias]
    record = capture(protocol)
    row = record["native"] if kind == "native" else record["generated_cpp"][0]
    assert row["path"] == str(target) and alias.is_symlink()
    assert assemble_origin(protocol, record)["evidence"] == record


def test_counterexample_duplicate_explicit_cpp_inventory_is_silently_deduplicated(protocol):
    protocol.artifact.program.generated_sources = [protocol.cpp, protocol.cpp]
    record = capture(protocol)
    assert record["generated_cpp"] == [helper._owner_file(protocol.cpp)]
    assert assemble_origin(protocol, record)["evidence"] == record
    duplicated = copy.deepcopy(record)
    duplicated["generated_cpp"] *= 2
    with pytest.raises(ValueError, match="inventory differs"):
        assemble_origin(protocol, duplicated)


def test_foreign_explicit_cpp_can_be_captured_but_strict_runtime_root_blocks_assembly(protocol, tmp_path):
    foreign = tmp_path / "foreign.cpp"
    foreign.write_bytes(b"FOREIGN OPAQUE CPP")
    protocol.artifact.program.generated_sources = [foreign]
    record = capture(protocol)
    assert record["generated_cpp"][0]["path"] == str(foreign)
    with pytest.raises(ValueError, match="escapes or aliases"):
        assemble_origin(protocol, record)


def test_no_generated_cpp_is_inferred_from_sibling_or_installed_hash(protocol):
    protocol.artifact.program.generated_sources = []
    assert protocol.cpp.exists()
    assert capture(protocol)["generated_cpp"] is None


def test_sidecar_schema_cannot_itself_be_used_as_external_root_approval(protocol):
    record = capture(protocol)
    path = protocol.roots["archive"] / "metadata.json"
    path.write_text(json.dumps(record))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="ROOT has not approved this exact pending template"):
        assembler.approved_pins(path, path, digest)
