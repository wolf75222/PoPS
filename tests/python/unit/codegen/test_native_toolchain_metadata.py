"""Declared SDK metadata admission and immutable native cache identity; no C++ build."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True, params=("platform_only", "compiler_bundle"))
def compiler_authority(request, tmp_path, monkeypatch):
    """Exercise both supported captures without depending on the runner's SDK."""
    from pops.codegen import _compiler_include_contract as compiler_contract
    from pops.codegen._compiler_include_tree import file_digest, snapshot_include_tree

    capture = None
    if request.param == "compiler_bundle":
        compiler = tmp_path / "configured-compiler"
        compiler.write_bytes(b"test configured compiler")
        root = tmp_path / "configured-includes"
        root.mkdir()
        (root / "vector").write_bytes(b"test configured header")
        size, digest = file_digest(str(compiler))
        tree = snapshot_include_tree(str(root))
        capture = compiler_contract.CapturedCompilerIncludes(
            str(compiler), str(compiler), size, digest,
            ((tree["root"], tree["sha256"], tree["file_count"]),),
        )
        assert capture.verify() == frozenset((str(root / "vector"),))
    monkeypatch.setattr(compiler_contract, "capture_compiler_includes", lambda: capture)
    return capture


def _sdk(capture):
    from pops.codegen._native_toolchain import CapturedNativeToolchainBundle

    return capture.sdk if type(capture) is CapturedNativeToolchainBundle else capture


@pytest.fixture
def declared_sdk(tmp_path, monkeypatch):
    from pops.codegen import _native_toolchain as toolchain

    root = tmp_path / "SDK with spaces"
    root.mkdir()
    metadata = root / "SDKSettings.json"
    metadata.write_bytes(b'{"Version":"27.0"}\n')
    monkeypatch.setattr(toolchain, "_platform_name", lambda: "darwin")
    return root, metadata


def _capture(root):
    from pops.codegen._native_toolchain import capture_native_toolchain

    return capture_native_toolchain({"SDKROOT": str(root)})


def _depfile(tmp_path, dependencies):
    source = tmp_path / "generated.cpp"
    source.write_text("// generated\n")
    depfile = tmp_path / "generated.d"
    paths = [source, *dependencies]
    depfile.write_text("generated.o: " + " ".join(
        str(path).replace(" ", "\\ ") for path in paths
    ) + "\n")
    return source, depfile


def _verify(tmp_path, monkeypatch, capture, dependencies):
    from pops.codegen import toolchain
    from pops.native_components import verify_prepared_native_dependencies

    monkeypatch.setattr(toolchain, "pops_authenticated_header_paths", lambda root: frozenset())
    source, depfile = _depfile(tmp_path, dependencies)
    return verify_prepared_native_dependencies(
        depfile, generated_source=source, pops_include_root=tmp_path,
        staged_components=(), toolchain_include_roots=(), toolchain_metadata=capture,
    )


def test_declared_metadata_admits_only_exact_file(declared_sdk, tmp_path, monkeypatch):
    root, metadata = declared_sdk
    capture = _capture(root)
    assert str(metadata) in _verify(tmp_path, monkeypatch, capture, [metadata])
    with pytest.raises(FrozenInstanceError):
        _sdk(capture).sha256 = "0" * 64


@pytest.mark.parametrize("declared", (None, "", "macosx", "relative/sdk"))
def test_darwin_requires_explicit_absolute_sdkroot(declared_sdk, declared):
    from pops.codegen._native_toolchain import capture_native_toolchain

    with pytest.raises(ValueError, match="explicitly declared absolute SDKROOT"):
        capture_native_toolchain({} if declared is None else {"SDKROOT": declared})


def test_linux_retains_existing_flags_without_sdkroot(monkeypatch, compiler_authority):
    from pops.codegen import _native_toolchain as toolchain

    monkeypatch.setattr(toolchain, "_platform_name", lambda: "linux")
    for environment in ({}, {"SDKROOT": "/ignored"}):
        capture = toolchain.capture_native_toolchain(environment)
        assert _sdk(capture) is None
        if compiler_authority is not None:
            assert capture.compiler is compiler_authority
            assert capture.verify() == compiler_authority.verify()
        else:
            assert capture is None
    flags = ["-I", "/closed/include", "-DPOPS_HAS_MPI"]
    assert toolchain.native_toolchain_flags(None, flags) == flags
    assert toolchain.verify_native_toolchain(None) == frozenset()


def test_declared_sdk_alias_is_pinned_canonically(declared_sdk, tmp_path):
    from pops.codegen._native_toolchain import native_toolchain_flags

    root, _ = declared_sdk
    alias = tmp_path / "SDK alias"
    alias.symlink_to(root, target_is_directory=True)
    capture = _capture(alias)
    assert _sdk(capture).sysroot == str(root)
    assert native_toolchain_flags(capture, ["-DPOPS_HAS_MPI"]) == [
        "-DPOPS_HAS_MPI", "-isysroot", str(root)
    ]
    assert native_toolchain_flags(capture, ["--sysroot=" + str(alias)]) == [
        "-isysroot", str(root)
    ]


@pytest.mark.parametrize("flags", (
    ["-isysroot"], ["-isysroot", "/unowned"], ["--sysroot=/unowned"],
    ["-isysroot/unowned"], ["-isysroot", "relative"],
    ["-Xclang=-isysroot", "-Xclang=/unowned"],
    ["-Xclang", "-isysroot", "-Xclang", "/unowned"],
    ["-Xclang=@/unowned-response-file"],
))
def test_conflicting_or_missing_sysroot_flags_are_rejected(declared_sdk, flags):
    from pops.codegen._native_toolchain import native_toolchain_flags

    capture = _capture(declared_sdk[0])
    with pytest.raises(ValueError, match="sysroot"):
        native_toolchain_flags(capture, flags)


def test_undeclared_or_neighboring_metadata_remains_rejected(
    declared_sdk, tmp_path, monkeypatch,
):
    root, metadata = declared_sdk
    capture = _capture(root)
    neighboring = root / "not-declared.hpp"
    neighboring.write_bytes(b"// unowned\n")
    other_sdk = tmp_path / "other SDK"
    other_sdk.mkdir()
    other_metadata = other_sdk / "SDKSettings.json"
    other_metadata.write_bytes(metadata.read_bytes())
    for proof, observed in ((None, metadata), (capture, neighboring), (capture, other_metadata)):
        with pytest.raises(RuntimeError, match="outside the authenticated"):
            _verify(tmp_path, monkeypatch, proof, [observed])


@pytest.mark.parametrize("change", ("bytes", "deleted", "symlink"))
def test_metadata_revalidated_after_capture(declared_sdk, tmp_path, monkeypatch, change):
    root, metadata = declared_sdk
    capture = _capture(root)
    if change == "bytes":
        metadata.write_bytes(b'{"Version":"28.0"}\n')
    else:
        metadata.unlink()
        if change == "symlink":
            outside = tmp_path / "foreign-metadata.json"
            outside.write_bytes(b'{"Version":"27.0"}\n')
            metadata.symlink_to(outside)
    with pytest.raises(RuntimeError, match="native SDK metadata"):
        _verify(tmp_path, monkeypatch, capture, [metadata])


def test_metadata_symlink_is_not_a_declared_regular_file(declared_sdk, tmp_path):
    root, metadata = declared_sdk
    raw = metadata.read_bytes()
    metadata.unlink()
    outside = tmp_path / "foreign-metadata.json"
    outside.write_bytes(raw)
    metadata.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        _capture(root)


def test_metadata_change_alters_program_artifact_identity(declared_sdk, monkeypatch):
    from pops.codegen import _artifact_identity as identity, toolchain

    root, metadata = declared_sdk
    first = _capture(root)
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "fake-native")
    graph = SimpleNamespace(graph_hash="graph")
    program = SimpleNamespace(name="unchanged physics")
    from pops.identity import make_identity
    semantic = make_identity("semantic", {"fixture": "unchanged physics"})
    monkeypatch.setattr("pops.identity.semantic.semantic_identity_of", lambda **kwargs: semantic)
    def artifact(capture):
        return identity.program_artifact_spec(
            snapshot=object(), model_authority=None, program=program, program_graph=graph,
            target="system", abi_key="unchanged ABI", compiler="c++", standard="c++20",
            source="// unchanged\n", cflags=[], lflags=[], optflags=[], libraries=(),
            native_dimension=2, toolchain_capture=capture,
        )[1]
    first_identity = artifact(first)
    metadata.write_bytes(b'{"Version":"28.0"}\n')
    second = _capture(root)
    assert _sdk(first).sha256 != _sdk(second).sha256
    assert first_identity != artifact(second)
    with pytest.raises(RuntimeError, match="changed after capture"):
        artifact(first)


def test_metadata_change_alters_model_artifact_identity(declared_sdk, monkeypatch):
    from pops.codegen import _artifact_identity as identity, _compile_emit, toolchain

    root, metadata = declared_sdk
    first = _capture(root)
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "fake-native")
    monkeypatch.setattr(_compile_emit, "model_hash", lambda model: "unchanged model")
    def artifact(capture):
        return identity.model_artifact_spec(
            SimpleNamespace(_ws_jacobian={}), backend="production", target="system", name="same",
            compiler="c++", standard="c++20", abi_key="unchanged ABI", hoist_reciprocals=False,
            toolchain_capture=capture,
        )[1]
    first_identity = artifact(first)
    metadata.write_bytes(b'{"Version":"28.0"}\n')
    assert first_identity != artifact(_capture(root))
    with pytest.raises(RuntimeError, match="changed after capture"):
        artifact(first)


@pytest.mark.parametrize("explicit_output", (False, True))
def test_metadata_changed_during_program_compile_publishes_nothing(
    declared_sdk, tmp_path, monkeypatch, explicit_output,
):
    from pops.codegen import _compile_drivers as drivers, toolchain
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    root, metadata = declared_sdk
    monkeypatch.setenv("SDKROOT", str(root))
    monkeypatch.setenv("POPS_CODEGEN_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(drivers, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(drivers, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, std: std or "c++20")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "TEST-FEATURES")
    monkeypatch.setattr("pops._native_selector.select_native_dimension", lambda dimension: None)
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    def compiler(command, where):
        assert command[command.index("-isysroot") + 1] == str(root)
        output = Path(command[command.index("-o") + 1])
        depfile = Path(command[command.index("-MF") + 1])
        generated = next(value for value in command if value.endswith("problem.cpp"))
        output.write_bytes(b"UNVERIFIED-BINARY")
        depfile.write_text("output: " + " ".join(
            value.replace(" ", "\\ ") for value in (generated, str(metadata))
        ) + "\n")
        metadata.write_bytes(b'{"Version":"changed during compile"}\n')
    monkeypatch.setattr(drivers, "_run_compile", compiler)
    program, model = _program_fixture("sdk_metadata_rejected")
    with pytest.raises(RuntimeError, match="changed after capture"):
        drivers.compile_problem(
            model=model, time=program, force=True, include=INCLUDE,
            so_path=str(tmp_path / "explicit.so") if explicit_output else None,
            native_dimension=2,
        )
    assert not tuple(tmp_path.rglob("*.so"))
    assert not tuple(tmp_path.rglob("*.dylib"))
    assert not tuple(tmp_path.rglob("*.pops-artifact.json"))


@pytest.mark.parametrize("explicit_output", (False, True))
def test_model_metadata_rechecked_before_publication(
    declared_sdk, tmp_path, monkeypatch, explicit_output,
):
    from pops.codegen import _compile_drivers as drivers, _compile_emit, toolchain

    root, metadata = declared_sdk
    monkeypatch.setenv("SDKROOT", str(root))
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(drivers, "_native_kokkos_compiler", lambda cxx: "c++")
    monkeypatch.setattr(drivers, "_abi_key_python", lambda *args: "unchanged ABI")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "TEST-FEATURES")
    monkeypatch.setattr(_compile_emit, "model_hash", lambda model: "unchanged model")
    model = SimpleNamespace(_elliptic_fields={}, _ws_jacobian={},
                            _check_require_metadata=lambda *args: None)
    def compiler(model, output, *args, **kwargs):
        capture = kwargs["_toolchain_capture"]
        assert _sdk(capture).sysroot == str(root)
        Path(output).write_bytes(b"UNVERIFIED-BINARY")
        metadata.write_bytes(b'{"Version":"changed during compile"}\n')
        return output
    monkeypatch.setattr(drivers, "compile_native", compiler)
    with pytest.raises(RuntimeError, match="changed after capture"):
        drivers.compile_model(
            model, include="mock-headers", std="c++20",
            so_path=str(tmp_path / "explicit.so") if explicit_output else None,
        )
    assert not tuple(tmp_path.rglob("*.so"))
    assert not tuple(tmp_path.rglob("*.pops-artifact.json"))


def test_model_metadata_rechecked_before_cached_artifact_return(
    declared_sdk, tmp_path, monkeypatch,
):
    from pops.codegen import _compile_drivers as drivers, _compile_emit, toolchain

    root, metadata = declared_sdk
    monkeypatch.setenv("SDKROOT", str(root))
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(drivers, "_native_kokkos_compiler", lambda cxx: "c++")
    monkeypatch.setattr(drivers, "_abi_key_python", lambda *args: "unchanged ABI")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "TEST-FEATURES")
    monkeypatch.setattr(_compile_emit, "model_hash", lambda model: "unchanged model")
    model = SimpleNamespace(_elliptic_fields={}, _ws_jacobian={},
                            _check_require_metadata=lambda *args: None)
    def compiler(model, output, *args, **kwargs):
        Path(output).write_bytes(b"MOCK-CACHE-BINARY")
        return output
    monkeypatch.setattr(drivers, "compile_native", compiler)
    output = drivers.compile_model(model, include="mock-headers", std="c++20")
    verify_cache = drivers.verify_cached_artifact
    def changed_after_cache_verification(*args, **kwargs):
        result = verify_cache(*args, **kwargs)
        metadata.write_bytes(b'{"Version":"changed during cache lookup"}\n')
        return result
    monkeypatch.setattr(drivers, "verify_cached_artifact", changed_after_cache_verification)
    with pytest.raises(RuntimeError, match="changed after capture"):
        drivers.compile_model(model, include="mock-headers", std="c++20")
    assert Path(output).read_bytes() == b"MOCK-CACHE-BINARY"


@pytest.mark.parametrize("route", ("model", "program"))
@pytest.mark.parametrize("override", (
    ["--sysroot=/unowned-last-driver-option"],
    ["-Xclang=-isysroot", "-Xclang=/unowned-forwarded-option"],
))
def test_link_flags_cannot_override_the_declared_compiler_sysroot(
    declared_sdk, tmp_path, monkeypatch, route, override,
):
    from pops.codegen import _compile_drivers as drivers, toolchain
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    root, _ = declared_sdk
    monkeypatch.setenv("SDKROOT", str(root))
    monkeypatch.setattr(drivers, "pops_loader_build_flags", lambda cxx=None: (
        "c++", [], override
    ))
    monkeypatch.setattr(drivers, "_check_headers_match_module", lambda include: "MOCKSIG")
    monkeypatch.setattr(drivers, "_warn_kokkos_parity", lambda: None)
    monkeypatch.setattr(drivers, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, std: std or "c++20")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "TEST-FEATURES")
    monkeypatch.setattr("pops._native_selector.select_native_dimension", lambda dimension: None)
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    program, model = _program_fixture("sdk_selector_rejected")
    if route == "model":
        monkeypatch.setattr(drivers, "emit_cpp_native_loader", lambda *args, **kwargs: "// generated")
        monkeypatch.setattr("pops.codegen.native_build.model_native_components", lambda model: ())
        monkeypatch.setattr(drivers, "model_hash", lambda model: "unchanged model")
        def invoke():
            return drivers.compile_native(model, str(tmp_path / "model.so"), include=INCLUDE)
    else:
        def invoke():
            return drivers.compile_problem(
                model=model, time=program, force=True, include=INCLUDE, native_dimension=2
            )
    with pytest.raises(ValueError, match="sysroot"):
        invoke()
    assert not tuple(tmp_path.rglob("*.so"))
    assert not tuple(tmp_path.rglob("*.pops-artifact.json"))
