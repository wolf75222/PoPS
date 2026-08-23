"""Command-shape contract for Kokkos ``nvcc_wrapper`` native loaders.

The wrapper forwards a split ``-I``, ``directory`` pair incorrectly to its host compiler.  These
tests only capture generated command lists; no compiler or generated binary is executed.
"""
from __future__ import annotations

from pathlib import Path


def test_native_loader_include_flags_join_only_for_nvcc_wrapper() -> None:
    from pops.codegen.toolchain import native_loader_include_flags

    flags = ["-DPOPS_HAS_KOKKOS", "-I", "/kokkos/include", "-I", "/mpi/include"]

    assert native_loader_include_flags("/opt/kokkos/bin/nvcc_wrapper", flags) == [
        "-DPOPS_HAS_KOKKOS",
        "-I/kokkos/include",
        "-I/mpi/include",
    ]
    assert native_loader_include_flags("/usr/bin/c++", flags) == flags


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
            ["-DPOPS_HAS_KOKKOS", "-I", "/kokkos/include", "-I", "/mpi/include"],
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
    output_index = command.index("-o")
    assert command[output_index - 1].endswith("model_native.cpp")
    assert command[output_index + 1] == str(output)
    assert command[output_index + 2 :] == ["-ldl", "-pthread"]


def test_aot_driver_normalizes_staged_component_include_flags() -> None:
    from pops.codegen import _compile_drivers as drivers

    source = Path(drivers.__file__).read_text()

    assert "native_loader_include_flags(\n                    cc," in source
    assert '[*flags, "-I", include, *component_include_flags]' in source
