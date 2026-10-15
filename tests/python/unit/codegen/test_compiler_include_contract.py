"""Compiler-declared file authority, independent of observed generated dependencies."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _encode(data):
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@pytest.fixture
def compiler_tree(tmp_path, monkeypatch):
    from pops.codegen import _native_toolchain, toolchain
    from pops.codegen._compiler_include_tree import file_digest, snapshot_include_tree

    compiler = tmp_path / "compiler"
    compiler.write_bytes(b"configured compiler\n")
    root = tmp_path / "implicit include"
    root.mkdir()
    header = root / "vector"
    header.write_bytes(b"configured standard header\n")
    size, digest = file_digest(str(compiler))
    tree = snapshot_include_tree(str(root))
    contract = {"schema_version": 1,
                "compiler": {"selector": str(compiler), "path": str(compiler),
                             "size": size, "sha256": digest},
                "include_trees": [{key: tree[key] for key in ("root", "sha256", "file_count")}]}
    module = SimpleNamespace(__cxx_compiler__=str(compiler),
                             __compiler_include_contract__=_encode(contract))
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    monkeypatch.setattr(_native_toolchain, "_platform_name", lambda: "linux")
    return module, contract, root, header


def _capture():
    from pops.codegen._native_toolchain import capture_native_toolchain

    return capture_native_toolchain({})


def _verify(tmp_path, monkeypatch, capture, dependencies):
    from pops.codegen import toolchain
    from pops.native_components import verify_prepared_native_dependencies

    monkeypatch.setattr(toolchain, "pops_authenticated_header_paths", lambda root: frozenset())
    source = tmp_path / "generated.cpp"
    source.write_text("// generated equation\n")
    depfile = tmp_path / "generated.d"
    depfile.write_text("generated.o: " + " ".join(str(path).replace(" ", "\\ ")
                       for path in (source, *dependencies)) + "\n")
    return verify_prepared_native_dependencies(
        depfile, generated_source=source, pops_include_root=tmp_path,
        staged_components=(), toolchain_include_roots=(), toolchain_metadata=capture)


def test_only_captured_regular_files_are_authorized(compiler_tree, tmp_path, monkeypatch):
    _, _, root, header = compiler_tree
    capture = _capture()
    assert capture.verify() == frozenset((str(header),))
    assert str(header) in _verify(tmp_path, monkeypatch, capture, (header,))
    neighboring = root.parent / "unowned.hpp"
    neighboring.write_bytes(header.read_bytes())
    with pytest.raises(RuntimeError, match="outside the authenticated"):
        _verify(tmp_path, monkeypatch, capture, (neighboring,))


@pytest.mark.parametrize("mutation", ("bytes", "neighbor", "delete", "symlink"))
def test_tree_changes_reject_before_publication(compiler_tree, tmp_path, monkeypatch, mutation):
    _, _, root, header = compiler_tree
    capture = _capture()
    if mutation == "bytes":
        header.write_bytes(b"changed content, same authority name")
    elif mutation == "neighbor":
        (root / "newly injected header").write_bytes(b"must not gain prefix authority")
    else:
        old = header.read_bytes()
        header.unlink()
        if mutation == "symlink":
            outside = tmp_path / "different source"
            outside.write_bytes(old)
            header.symlink_to(outside)
    with pytest.raises(RuntimeError, match="include tree changed"):
        _verify(tmp_path, monkeypatch, capture, (header,))


def test_changed_compiler_rejects_even_unchanged_headers(compiler_tree):
    module, _, _, _ = compiler_tree
    capture = _capture()
    Path(module.__cxx_compiler__).write_bytes(b"different compiler")
    with pytest.raises(RuntimeError, match="compiler changed"):
        capture.verify()


def test_compiler_selector_symlink_cannot_change_after_capture(compiler_tree, tmp_path):
    module, contract, _, _ = compiler_tree
    alias = tmp_path / "compiler selector"
    alias.symlink_to(module.__cxx_compiler__)
    module.__cxx_compiler__ = str(alias)
    contract["compiler"]["selector"] = str(alias)
    module.__compiler_include_contract__ = _encode(contract)
    capture = _capture()
    alias.unlink()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(Path(capture.compiler.compiler_path).read_bytes())
    alias.symlink_to(replacement)
    with pytest.raises(RuntimeError, match="compiler changed"):
        capture.verify()


@pytest.mark.parametrize("mutation", ("schema", "compiler", "hash", "count", "duplicate", "field"))
def test_altered_native_contract_rejected(compiler_tree, mutation):
    module, contract, _, _ = compiler_tree
    if mutation == "schema":
        contract["schema_version"] = True
    elif mutation == "compiler":
        contract["compiler"]["path"] += ".foreign"
    elif mutation == "hash":
        contract["include_trees"][0]["sha256"] = "0" * 64
    elif mutation == "count":
        contract["include_trees"][0]["file_count"] = True
    elif mutation == "duplicate":
        contract["include_trees"] *= 2
    else:
        contract["ambient_root"] = "/"
    module.__compiler_include_contract__ = _encode(contract)
    with pytest.raises((ValueError, RuntimeError)):
        _capture()


def test_canonical_contract_rejects_duplicate_json_keys(compiler_tree):
    module, _, _, _ = compiler_tree
    module.__compiler_include_contract__ = module.__compiler_include_contract__.replace(
        '"schema_version":1', '"schema_version":0,"schema_version":1')
    with pytest.raises(ValueError, match="canonical JSON"):
        _capture()


def test_old_native_grants_no_implicit_file_authority(compiler_tree):
    module, _, _, _ = compiler_tree
    del module.__compiler_include_contract__
    assert _capture() is None


def test_declared_symlink_tree_is_finite_and_target_authenticated(tmp_path):
    from pops.codegen._compiler_include_tree import snapshot_include_tree

    root = tmp_path / "includes"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    header = target / "value.hpp"
    header.write_text("// exact bytes\n")
    (root / "alias").symlink_to(target, target_is_directory=True)
    (target / "cycle").symlink_to(root, target_is_directory=True)
    first = snapshot_include_tree(str(root))
    assert first["files"] == frozenset((str(header),))
    assert first["file_count"] == 1
    assert first == snapshot_include_tree(str(root))
    (root / "alias").unlink()
    assert first["sha256"] != snapshot_include_tree(str(root))["sha256"]


def test_native_compiler_capture_preserves_flags(compiler_tree):
    from pops.codegen._native_toolchain import native_toolchain_flags

    flags = ["-std=c++20", "-I", "/existing/explicit/include"]
    assert native_toolchain_flags(_capture(), flags) == flags


def test_platform_and_compiler_authorities_compose(compiler_tree, tmp_path, monkeypatch):
    from pops.codegen import _native_toolchain

    _, _, _, header = compiler_tree
    sdk = tmp_path / "platform sdk"
    sdk.mkdir()
    metadata = sdk / "SDKSettings.json"
    metadata.write_text('{"Version":"1"}\n')
    monkeypatch.setattr(_native_toolchain, "_platform_name", lambda: "darwin")
    capture = _native_toolchain.capture_native_toolchain({"SDKROOT": str(sdk)})
    assert capture.verify() == frozenset((str(header), str(metadata)))
    assert _native_toolchain.native_toolchain_flags(capture, []) == ["-isysroot", str(sdk)]
    assert capture.to_data()["compiler"]["contract"] == "pops.native-compiler-includes@1"


def test_compiler_capture_changes_artifact_identity(compiler_tree, monkeypatch):
    from pops.codegen import _artifact_identity as identity, toolchain
    from pops.codegen._compiler_include_tree import snapshot_include_tree
    from pops.identity import make_identity

    module, contract, root, header = compiler_tree
    first = _capture()
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "native")
    semantic = make_identity("semantic", {"fixture": "same equations"})
    monkeypatch.setattr("pops.identity.semantic.semantic_identity_of", lambda **kwargs: semantic)
    def artifact(capture):
        return identity.program_artifact_spec(
            snapshot=object(), model_authority=None, program=SimpleNamespace(name="same"),
            program_graph=SimpleNamespace(graph_hash="graph"), target="system", abi_key="ABI",
            compiler="c++", standard="c++20", source="// same equations\n", cflags=[],
            lflags=[], optflags=[], libraries=(), native_dimension=2, toolchain_capture=capture)[1]
    first_identity = artifact(first)
    header.write_bytes(b"next native build's standard library")
    tree = snapshot_include_tree(str(root))
    contract["include_trees"] = [{key: tree[key] for key in ("root", "sha256", "file_count")}]
    module.__compiler_include_contract__ = _encode(contract)
    assert first_identity != artifact(_capture())
    with pytest.raises(RuntimeError, match="changed after"):
        artifact(first)


@pytest.mark.parametrize("explicit_output", (False, True))
def test_tree_changed_during_compile_publishes_no_artifact(
    compiler_tree, tmp_path, monkeypatch, explicit_output,
):
    from pops.codegen import _compile_drivers as drivers, toolchain
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    _, _, _, header = compiler_tree
    monkeypatch.setenv("POPS_CODEGEN_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(drivers, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(drivers, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, std: std or "c++20")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "TEST-FEATURES")
    monkeypatch.setattr("pops._native_selector.select_native_dimension", lambda dimension: None)
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    def compiler(command, where):
        output = Path(command[command.index("-o") + 1])
        depfile = Path(command[command.index("-MF") + 1])
        generated = next(value for value in command if value.endswith("problem.cpp"))
        output.write_bytes(b"UNVERIFIED-BINARY")
        depfile.write_text("output: " + " ".join(value.replace(" ", "\\ ")
                           for value in (generated, str(header))) + "\n")
        header.write_bytes(b"changed after compiler captured this header")
    monkeypatch.setattr(drivers, "_run_compile", compiler)
    program, model = _program_fixture("compiler_tree_rejected")
    with pytest.raises(RuntimeError, match="include tree changed"):
        drivers.compile_problem(model=model, time=program, force=True, include=INCLUDE,
            so_path=str(tmp_path / "explicit.so") if explicit_output else None, native_dimension=2)
    assert not tuple(tmp_path.rglob("*.so"))
    assert not tuple(tmp_path.rglob("*.dylib"))
    assert not tuple(tmp_path.rglob("*.pops-artifact.json"))


def test_build_generator_and_runtime_share_exact_contract(compiler_tree, tmp_path):
    import subprocess
    import sys

    module, _, root, header = compiler_tree
    script = Path(__file__).resolve().parents[4] / "scripts/write_compiler_include_contract.py"
    output = tmp_path / "compiler_contract.hpp"
    command = [sys.executable, str(script), "--compiler", module.__cxx_compiler__,
               "--output", str(output), "--root", str(root)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["files"] == 1
    text = output.read_text()
    module.__compiler_include_contract__ = text.split('R"POPS_TC(', 1)[1].split(')POPS_TC"', 1)[0]
    assert _capture().verify() == frozenset((str(header),))
    before = output.stat().st_mtime_ns
    subprocess.run(command, text=True, capture_output=True, check=True)
    assert output.stat().st_mtime_ns == before
