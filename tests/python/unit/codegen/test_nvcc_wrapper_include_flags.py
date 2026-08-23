"""Command-shape contract for Kokkos ``nvcc_wrapper`` native loaders.

The wrapper forwards a split ``-I``, ``directory`` pair incorrectly to its host compiler.  These
tests only capture generated command lists; no compiler or generated binary is executed.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_native_loader_include_flags_join_only_for_nvcc_wrapper() -> None:
    from pops.codegen.toolchain import (
        _pops_nvcc_wrapper_compile_environment,
        _pops_nvcc_wrapper_compile_flags,
        native_loader_codegen_key,
        native_loader_include_flags,
    )

    flags = ["-DPOPS_HAS_KOKKOS", "-I", "/kokkos/include", "-I", "/mpi/include"]

    assert native_loader_include_flags("/opt/kokkos/bin/nvcc_wrapper", flags) == [
        "-DPOPS_HAS_KOKKOS",
        "-I/kokkos/include",
        "-I/mpi/include",
    ]
    assert native_loader_include_flags("/usr/bin/c++", flags) == flags
    assert _pops_nvcc_wrapper_compile_flags("/opt/kokkos/bin/nvcc_wrapper") == [
        "--expt-relaxed-constexpr",
    ]
    assert _pops_nvcc_wrapper_compile_flags("/usr/bin/c++") == []
    assert _pops_nvcc_wrapper_compile_environment("/opt/kokkos/bin/nvcc_wrapper") == {
        "NVCC_PREPEND_FLAGS": "--split-compile=2"
    }
    assert _pops_nvcc_wrapper_compile_environment("/usr/bin/c++") == {}
    assert native_loader_codegen_key("/opt/kokkos/bin/nvcc_wrapper").endswith(
        "--expt-relaxed-constexpr,NVCC_PREPEND_FLAGS=--split-compile=2"
    )
    assert native_loader_codegen_key("/usr/bin/c++").endswith(":host")


def test_compile_native_nvcc_wrapper_joins_kokkos_mpi_and_sdk_includes(monkeypatch, tmp_path) -> None:
    from pops.codegen import _compile_drivers as drivers

    captured: list[tuple[list[str], str]] = []
    monkeypatch.setattr(drivers, "_check_headers_match_module", lambda include: "signature")
    monkeypatch.setattr(drivers, "_warn_kokkos_parity", lambda: None)
    monkeypatch.setattr(drivers, "emit_cpp_native_loader", lambda *args, **kwargs: "// native")
    monkeypatch.setattr(
        drivers,
        "pops_loader_build_flags",
        lambda cxx: (
            "/opt/kokkos/bin/nvcc_wrapper",
            [
                "-DPOPS_HAS_KOKKOS",
                "-I",
                "/kokkos/include",
                "-I",
                "/mpi/include",
                "-extended-lambda",
                "-Wext-lambda-captures-this",
                "-arch=sm_90",
                "--expt-relaxed-constexpr",
            ],
            ["-ldl", "-pthread"],
        ),
    )
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, standard: standard)
    monkeypatch.setattr(drivers, "_dsl_optflags", lambda: ["-O3"])
    monkeypatch.setattr(
        drivers,
        "_run_compile",
        lambda command, what: captured.append((list(command), what)),
    )

    output = tmp_path / "model.so"
    drivers.compile_native(
        object(),
        str(output),
        include="/pops/include",
        cxx="ignored",
        model_identity="model",
    )

    assert len(captured) == 1
    command, what = captured[0]
    assert what == "backend production, compile_native"
    assert command[0] == "/opt/kokkos/bin/nvcc_wrapper"
    assert "-I" not in command
    assert command.index("-I/kokkos/include") < command.index("-I/mpi/include")
    assert command.index("-I/mpi/include") < command.index("-I/pops/include")
    assert command.index("-extended-lambda") < command.index("-arch=sm_90")
    assert command.index("-arch=sm_90") < command.index("-I/pops/include")
    assert command.count("--expt-relaxed-constexpr") == 1
    output_index = command.index("-o")
    assert command.index("--expt-relaxed-constexpr") < output_index
    assert "--split-compile=2" not in command
    assert command[output_index - 1].endswith("model_native.cpp")
    assert command[output_index + 1] == str(output)
    assert command[output_index + 2 :] == ["-ldl", "-pthread"]


def test_compile_native_keeps_exact_failed_source_for_cuda_diagnosis(monkeypatch, tmp_path) -> None:
    from pops.codegen import _compile_drivers as drivers

    monkeypatch.setattr(drivers, "_check_headers_match_module", lambda include: "signature")
    monkeypatch.setattr(drivers, "_warn_kokkos_parity", lambda: None)
    monkeypatch.setattr(drivers, "emit_cpp_native_loader", lambda *args, **kwargs: "// native\n")
    monkeypatch.setattr(drivers, "pops_loader_build_flags", lambda cxx: ("nvcc_wrapper", [], []))
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, standard: standard)
    monkeypatch.setattr(drivers, "_dsl_optflags", lambda: ["-O3"])
    monkeypatch.setattr(drivers, "_run_compile", lambda *args: (_ for _ in ()).throw(
        RuntimeError("cicc died due to signal 11")
    ))

    output = tmp_path / "model.so"
    with pytest.raises(RuntimeError, match="model.failed.cpp") as failure:
        drivers.compile_native(
            object(), str(output), include="/pops/include", model_identity="native-failure"
        )

    failed = tmp_path / "model.failed.cpp"
    assert failed.read_text() == "// native\n"
    assert "cicc died due to signal 11" in str(failure.value)


def test_aot_driver_normalizes_staged_component_include_flags() -> None:
    from pops.codegen import _compile_drivers as drivers

    source = Path(drivers.__file__).read_text()

    assert "native_loader_include_flags(\n                    cc," in source
    assert '[*flags, "-I", include, *component_include_flags]' in source
