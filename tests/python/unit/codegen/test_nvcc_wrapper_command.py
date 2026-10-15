"""Compiler argv regression; GPU execution is qualified by a separate receipt."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.codegen.toolchain import normalize_native_compiler_command, _run_compile


def test_nvcc_wrapper_search_operands_remain_single_ordered_arguments():
    original = ["/kokkos/bin/nvcc_wrapper", "-shared", "-I", "/sdk/include",
                "-I", "/component with spaces/include", "-I/sdk/include",
                "-L", "/first/lib", "-l", "kokkoscore", "-L", "/second/lib",
                "-D", 'NAME="a,b"', "program.cpp", "-o", "program.so"]
    expected = ["/kokkos/bin/nvcc_wrapper", "-shared", "-I/sdk/include",
                "-I/component with spaces/include", "-I/sdk/include",
                "-L/first/lib", "-lkokkoscore", "-L/second/lib", '-DNAME="a,b"',
                "program.cpp", "-o", "program.so"]
    assert normalize_native_compiler_command(original) == expected
    assert normalize_native_compiler_command(expected) == expected
    assert original[2:4] == ["-I", "/sdk/include"]


def test_host_commands_and_opaque_forwarded_options_are_unchanged():
    host = ["/usr/bin/clang++", "-I", "/sdk/include", "program.cpp", "-o", "program.so"]
    assert normalize_native_compiler_command(host) == host
    opaque = ["nvcc_wrapper", "-Xcompiler", "-I", "-Xlinker", "-L",
              "-I", "/sdk", "--", "-I", "literal-source.cpp"]
    assert normalize_native_compiler_command(opaque) == [
        "nvcc_wrapper", "-Xcompiler", "-I", "-Xlinker", "-L", "-I/sdk",
        "--", "-I", "literal-source.cpp"]


@pytest.mark.parametrize("tail", [["-I"], ["-L", ""], ["-l"], ["-D", ""]])
def test_missing_wrapper_operand_is_rejected_before_execution(tail):
    with pytest.raises(ValueError, match="nonempty operand"):
        normalize_native_compiler_command(["nvcc_wrapper", *tail])


def test_execution_and_retained_command_can_share_exact_normalized_argv(monkeypatch):
    observed = []

    def run(command, **kwargs):
        observed.append(list(command))
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")

    monkeypatch.setattr("subprocess.run", run)
    command = ["nvcc_wrapper", "-I", "/sdk/include", "program.cpp", "-o", "program.so"]
    retained = normalize_native_compiler_command(command)
    _run_compile(retained, "test")
    _run_compile(command, "test")
    assert observed == [retained, retained]


def test_model_driver_records_the_same_command_it_executes(tmp_path, monkeypatch):
    from pops.codegen import _compile_drivers as drivers
    from pops.codegen import model_compile_evidence, native_build

    monkeypatch.setattr(drivers, "capture_native_toolchain", lambda: None)
    monkeypatch.setattr(drivers, "verify_native_toolchain", lambda _: None)
    monkeypatch.setattr(drivers, "_check_headers_match_module", lambda _: "test-sdk")
    monkeypatch.setattr(drivers, "_warn_kokkos_parity", lambda: None)
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda _, standard: standard)
    monkeypatch.setattr(drivers, "emit_cpp_native_loader", lambda *args, **kwargs: "// test TU\n")
    monkeypatch.setattr(drivers, "pops_loader_build_flags", lambda _: (
        "nvcc_wrapper", ["-I", "/kokkos/include"], ["-L", "/kokkos/lib", "-l", "kokkoscore"]))
    monkeypatch.setattr(drivers, "deterministic_component_link_flags", list)
    monkeypatch.setattr(native_build, "model_native_components", lambda _: ())
    monkeypatch.setattr(native_build, "stage_native_components", lambda *args: ([], {}, ()))
    recorded, executed = [], []
    monkeypatch.setattr(drivers, "_run_compile", lambda command, _: executed.append(list(command)))

    def retain(binary, source, command, signature, original):
        assert command.count(source) == 1
        assert signature == "test-sdk"
        recorded.append(list(command))
        original(command, "test")

    monkeypatch.setattr(model_compile_evidence, "retain", retain)
    drivers.compile_native(object(), str(tmp_path / "model.so"), include="/pops/include",
                           std="c++20", model_identity="test-model")
    assert recorded == executed
    assert all(option not in recorded[0] for option in ("-I", "-L", "-l"))
    assert "-I/pops/include" in recorded[0] and "-I/kokkos/include" in recorded[0]
    assert recorded[0][-2:] == ["-L/kokkos/lib", "-lkokkoscore"]


def test_cuda_command_policy_is_an_explicit_cache_contract():
    from pops.codegen.toolchain import native_compiler_command_contract

    assert native_compiler_command_contract("clang++") == {}
    assert native_compiler_command_contract("/kokkos/bin/nvcc_wrapper") == {
        "compiler_command_contract": "pops.nvcc-wrapper-joined-operands@1"}
