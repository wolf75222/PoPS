"""Path-independent link identity for generated Program plugins."""
from __future__ import annotations

from pops.codegen import compile_link_flags


def test_darwin_program_install_name_is_path_independent(monkeypatch):
    monkeypatch.setattr(compile_link_flags.sys, "platform", "darwin")

    flags = compile_link_flags.deterministic_program_link_flags(["-shared", "-lfoo"])

    assert flags == [
        "-shared",
        "-lfoo",
        "-Wl,-install_name,@rpath/pops_program.so",
    ]


def test_non_darwin_program_link_flags_are_unchanged(monkeypatch):
    monkeypatch.setattr(compile_link_flags.sys, "platform", "linux")

    assert compile_link_flags.deterministic_program_link_flags(("-shared", "-lfoo")) == [
        "-shared",
        "-lfoo",
    ]


def test_darwin_component_install_name_is_path_independent(monkeypatch):
    monkeypatch.setattr(compile_link_flags.sys, "platform", "darwin")
    original = ["-shared", "-lfoo"]

    assert compile_link_flags.deterministic_component_link_flags(original) == [
        "-shared", "-lfoo", "-Wl,-install_name,@rpath/pops_component.dylib",
    ]
    assert original == ["-shared", "-lfoo"]


def test_non_darwin_component_link_flags_are_unchanged(monkeypatch):
    monkeypatch.setattr(compile_link_flags.sys, "platform", "linux")

    assert compile_link_flags.deterministic_component_link_flags(("-shared", "-lfoo")) == [
        "-shared", "-lfoo",
    ]


def test_native_model_spec_invalidates_path_dependent_cached_binaries(monkeypatch):
    from pops.codegen import _artifact_identity, _compile_emit, cache, toolchain

    monkeypatch.setattr(compile_link_flags.sys, "platform", "darwin")
    monkeypatch.setattr(_compile_emit, "model_hash", lambda model: "model-hash")
    monkeypatch.setattr(cache, "_platform_cache_key", lambda: "fixed-platform")
    monkeypatch.setattr(cache, "_dsl_optflags", lambda: ["-O2"])
    monkeypatch.setattr(cache, "_precision_cache_key", lambda: "double")
    monkeypatch.setattr(cache, "_registry_cache_key", lambda: "fixed-registry")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "fixed-features")

    def identity():
        return _artifact_identity.model_artifact_spec(
            object(), backend="production", target="system", name="model",
            compiler="cxx", standard="c++20", abi_key="abi", hoist_reciprocals=False)

    new_semantic, new_spec = identity()
    monkeypatch.setattr(compile_link_flags, "deterministic_component_link_flags", list)
    old_semantic, old_spec = identity()
    assert new_semantic == old_semantic
    assert new_spec != old_spec
