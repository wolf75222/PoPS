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
