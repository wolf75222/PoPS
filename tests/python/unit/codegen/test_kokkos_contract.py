from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module(root: Path, *, core: bytes = b"core", config: bytes = b"config",
            compile_options=(), compile_definitions=(), link_options=(), compile_features=()):
    include = root / "include"
    include.mkdir(parents=True)
    paths = (include / "Kokkos_Core.hpp", include / "KokkosCore_config.h")
    paths[0].write_bytes(core)
    paths[1].write_bytes(config)
    hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)
    material = (
        "compiler=/host/cxx\nstandard=20\n"
        + "".join(f"{name}={value}\n" for name, values in (
            ("compile_option", compile_options), ("compile_definition", compile_definitions),
            ("link_option", link_options), ("compile_feature", compile_features)) for value in values)
        +
        f"include={include}\n"
        f"header={paths[0]};sha256={hashes[0]}\n"
        f"header={paths[1]};sha256={hashes[1]}\n"
    ).encode()
    return SimpleNamespace(
        __has_kokkos__=True,
        __kokkos_contract__={
            "schema_version": 2,
            "compiler": "/host/cxx",
            "standard": "20",
            "compile_options": tuple(compile_options),
            "compile_definitions": tuple(compile_definitions),
            "link_options": tuple(link_options),
            "compile_features": tuple(compile_features),
            "abi_sha256": hashlib.sha256(material).hexdigest(),
            "include_dirs": (str(include),),
            "header_paths": tuple(map(str, paths)),
            "header_sha256": hashes,
        },
    )


def _clear_overrides(monkeypatch):
    for name in ("POPS_KOKKOS_ROOT", "Kokkos_ROOT", "KOKKOS_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_baked_kokkos_contract_needs_no_shell_environment(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)

    root, includes, abi = toolchain._native_kokkos_selection()
    assert root == str(tmp_path / "built")
    assert includes == (str(tmp_path / "built" / "include"),)
    assert abi == module.__kokkos_contract__["abi_sha256"]

    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda _cxx: "c++")
    compile_flags, _ = toolchain._native_kokkos_flags()
    assert compile_flags[:4] == ["-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE", "-I", includes[0]]


def test_explicit_relocated_root_must_match_baked_headers(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _module(tmp_path / "relocated")
    shutil.rmtree(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("POPS_KOKKOS_ROOT", str(tmp_path / "relocated"))
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)

    root, includes, _ = toolchain._native_kokkos_selection()
    assert root == str(tmp_path / "relocated")
    assert includes == (str(tmp_path / "relocated" / "include"),)


@pytest.mark.parametrize("compiler", ["c++", "/cuda/bin/nvcc_wrapper"])
@pytest.mark.parametrize("config, enabled", [
    (b"#define KOKKOS_ENABLE_SERIAL\n", False),
    (b"#define KOKKOS_ENABLE_CUDA\n/* #undef KOKKOS_ENABLE_OPENMP */\n", False),
    (b"/*\n#define KOKKOS_ENABLE_OPENMP\n*/\n", False),
    (b"#define KOKKOS_ENABLE_OPENMP\n", True),
    (b"#define KOKKOS_ENABLE_CUDA\n#define KOKKOS_ENABLE_OPENMP 1\n", True),
])
def test_authenticated_openmp_backend_flags_apply_to_cuda_host(
        monkeypatch, tmp_path, compiler, config, enabled):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built", config=config)
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda _cxx: compiler)
    cflags, lflags = toolchain._native_kokkos_flags()
    assert cflags.count("-fopenmp") == int(enabled)
    assert lflags == ["-ldl", "-pthread", *(["-fopenmp"] if enabled else [])]
    assert all("kokkos" not in flag.lower() for flag in lflags)


def test_backend_selection_reauthenticates_captured_configuration(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built", config=b"#define KOKKOS_ENABLE_OPENMP\n")
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    includes = toolchain._native_kokkos_include_dirs()
    (Path(includes[0]) / "KokkosCore_config.h").write_text("#define KOKKOS_ENABLE_CUDA\n")
    with pytest.raises(RuntimeError, match="changed before backend flag selection"):
        toolchain._native_kokkos_openmp(includes)


def test_cuda_usage_requirements_are_replayed_in_order_and_authenticated(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    compile_options = ("-extended-lambda", "-expt-relaxed-constexpr", "-arch=sm_90", "-rdc=true",
                       "SHELL:-Xcompiler -fopenmp", "-Xcompiler", "-Wno-unused", "-Xcompiler", "-Wall")
    module = _module(tmp_path / "built", config=b"#define KOKKOS_ENABLE_CUDA\n#define KOKKOS_ENABLE_OPENMP\n",
                     compile_options=compile_options, compile_definitions=("KOKKOS_DEPENDENCE",),
                     link_options=("-arch=sm_90", "-rdc=true"), compile_features=("cxx_std_20",))
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    cflags, lflags = toolchain._native_kokkos_flags()
    assert cflags[4:] == ["-DKOKKOS_DEPENDENCE", *compile_options[:4], "-Xcompiler", "-fopenmp",
                          *compile_options[5:]]
    assert lflags == ["-ldl", "-pthread", "-arch=sm_90", "-rdc=true", "-fopenmp"]
    module.__kokkos_contract__["compile_options"] = ("-arch=sm_80",)
    with pytest.raises(RuntimeError, match="does not authenticate"):
        toolchain._native_kokkos_flags()


def test_header_only_legacy_contract_cannot_omit_cuda_usage_requirements(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    module.__kokkos_contract__["schema_version"] = 1
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    with pytest.raises(RuntimeError, match="schema_version"):
        toolchain._native_kokkos_contract()


def test_cmake_cuda_usage_contract_hash_matches_the_exported_python_payload(tmp_path):
    """Run the production normalizer and ABI builder, without configuring or compiling PoPS."""
    cmake = shutil.which("cmake")
    assert cmake is not None
    root = Path(__file__).resolve().parents[4]
    production = (root / "CMakeLists.txt").read_text()
    normalizer = re.search(r"function\(_pops_normalize_kokkos_replay_records\b.*?endfunction\(\)",
                           production, re.DOTALL)
    assert normalizer is not None
    start = production.index('set(_pops_kokkos_header_paths "")')
    end = production.index('list(JOIN _pops_kokkos_normalized_includes', start)
    module = _module(tmp_path / "built", compile_options=("-extended-lambda", "-arch=sm_90", "SHELL:-fopenmp"),
                     compile_definitions=("KOKKOS_DEPENDENCE",), link_options=("-arch=sm_90", "-rdc=true"),
                     compile_features=("cxx_std_20",))
    include = module.__kokkos_contract__["include_dirs"][0]
    script = tmp_path / "replay.cmake"
    script.write_text(
        'cmake_minimum_required(VERSION 3.21)\n' + normalizer.group(0) + '\n'
        'set(CMAKE_CXX_COMPILER "/host/cxx")\nset(CMAKE_CXX_STANDARD 20)\n'
        f'set(_pops_kokkos_normalized_includes "{include}")\n'
        '_pops_normalize_kokkos_replay_records("compile options" '
        '"$<$<COMPILE_LANGUAGE:CXX>:-extended-lambda;-arch=sm_90>;SHELL:-fopenmp" '
        '_pops_kokkos_normalized_compile_options)\n'
        '_pops_normalize_kokkos_replay_records("link options" '
        '"$<$<LINK_LANGUAGE:CXX>:-arch=sm_90;-rdc=true>" _pops_kokkos_normalized_link_options)\n'
        'set(_pops_kokkos_normalized_compile_definitions "KOKKOS_DEPENDENCE")\n'
        'set(_pops_kokkos_normalized_compile_features "cxx_std_20")\n'
        + production[start:end]
        + f'file(WRITE "{tmp_path / "abi.txt"}" "${{POPS_NATIVE_KOKKOS_ABI}}")\n')
    subprocess.run([cmake, "-P", str(script)], check=True, capture_output=True, text=True)
    assert (tmp_path / "abi.txt").read_text() == module.__kokkos_contract__["abi_sha256"]


def test_relocated_openmp_flags_preserve_macos_single_runtime(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    config = b"#define KOKKOS_ENABLE_OPENMP\n"
    module = _module(tmp_path / "built", config=config)
    _module(tmp_path / "relocated", config=config)
    shutil.rmtree(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("POPS_KOKKOS_ROOT", str(tmp_path / "relocated"))
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    monkeypatch.setattr(toolchain.sys, "platform", "darwin")
    monkeypatch.setattr(toolchain, "_libomp_prefix", lambda: "/host/libomp")
    cflags, lflags = toolchain._native_kokkos_flags()
    assert cflags[-4:] == ["-Xpreprocessor", "-fopenmp", "-I", "/host/libomp/include"]
    assert lflags == ["-ldl", "-pthread"]


def test_explicit_different_kokkos_root_is_rejected(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _module(tmp_path / "other", config=b"different backend")
    _clear_overrides(monkeypatch)
    monkeypatch.setenv("Kokkos_ROOT", str(tmp_path / "other"))
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)

    with pytest.raises(RuntimeError, match="differs from the installation"):
        toolchain._native_kokkos_selection()


def test_kokkos_contract_fails_closed_when_header_changes(monkeypatch, tmp_path):
    from pops.codegen import toolchain

    module = _module(tmp_path / "built")
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    (tmp_path / "built" / "include" / "KokkosCore_config.h").write_text("upgraded")

    with pytest.raises(RuntimeError, match="changed in place"):
        toolchain._native_kokkos_selection()


def test_cmake_selected_kokkos_precedes_a_dependency_prefix_with_foreign_headers(tmp_path):
    """Configure usage requirements only; no compiler, native build, or runtime fixture."""
    cmake = shutil.which("cmake")
    assert cmake is not None, "CMake is required to verify the native include contract"
    root = Path(__file__).resolve().parents[4]
    production = (root / "CMakeLists.txt").read_text()
    promotion = re.search(r'target_include_directories\(pops SYSTEM BEFORE INTERFACE\s+'
                          r'"\$<BUILD_INTERFACE:\$\{_pops_kokkos_normalized_includes\}>"\)', production)
    assert promotion is not None
    selected = tmp_path / "selected-kokkos"
    foreign = tmp_path / "dependency-prefix"
    selected.mkdir()
    foreign.mkdir()
    # Both roots deliberately expose the same header names, as in a conda
    # MPI/HDF5 prefix plus a separately installed CUDA Kokkos.
    for path in (selected, foreign):
        (path / "Kokkos_Core.hpp").write_text(path.name)
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(include_contract LANGUAGES NONE)\n"
        "add_library(Kokkos::kokkos INTERFACE IMPORTED)\n"
        f'set_target_properties(Kokkos::kokkos PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "{selected}")\n'
        "add_library(pops INTERFACE)\n"
        f'target_include_directories(pops INTERFACE "{foreign}")\n'
        "target_link_libraries(pops INTERFACE Kokkos::kokkos)\n"
        f'set(_pops_kokkos_normalized_includes "{selected}")\n'
        + promotion.group(0) + "\n"
        "add_library(consumer INTERFACE)\n"
        "target_link_libraries(consumer INTERFACE pops)\n"
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/includes.txt" '
        'CONTENT "$<TARGET_PROPERTY:consumer,INTERFACE_INCLUDE_DIRECTORIES>" TARGET consumer)\n')
    build = tmp_path / "configured"
    subprocess.run([cmake, "-S", str(source), "-B", str(build)], check=True,
                   capture_output=True, text=True)
    includes = (build / "includes.txt").read_text().split(";")
    assert includes.index(str(selected)) < includes.index(str(foreign))


@pytest.mark.parametrize("cuda,compiler,expected", [
    (True, "NVIDIA", ["--expt-relaxed-constexpr", "--Werror=cross-execution-space-call",
                      "SHELL:-Xcudafe --diag_error=20013"]),
    (False, "NVIDIA", []),
    (True, "Clang", []),
])
def test_cmake_device_language_contract_applies_to_sdk_and_generated_dso(
    tmp_path, monkeypatch, cuda, compiler, expected
):
    """Execute production CMake policy; replay and authenticate its exact options."""
    from pops.codegen import toolchain

    cmake = shutil.which("cmake")
    assert cmake is not None
    production = (Path(__file__).resolve().parents[4] / "CMakeLists.txt").read_text()
    policy = re.search(r"function\(_pops_cuda_language_options\b.*?endfunction\(\)",
                       production, re.DOTALL)
    assert policy is not None
    begin = production.index("_pops_cuda_language_options(_pops_cuda_options)")
    end = production.index("\n_pops_collect_kokkos_target_property(", begin)
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\nproject(cuda_contract LANGUAGES NONE)\n"
        "add_library(pops INTERFACE)\n"
        f"set(Kokkos_ENABLE_CUDA {'ON' if cuda else 'OFF'})\n"
        f'set(Kokkos_CXX_COMPILER_ID "{compiler}")\n'
        + policy.group(0) + "\n" + production[begin:end] + "\n"
        + 'get_target_property(_sdk pops INTERFACE_COMPILE_OPTIONS)\n'
        + 'file(WRITE "${CMAKE_BINARY_DIR}/sdk.txt" "${_sdk}")\n'
        + 'file(WRITE "${CMAKE_BINARY_DIR}/replay.txt" "${_pops_kokkos_compile_options}")\n'
    )
    build = tmp_path / "build"
    subprocess.run([cmake, "-S", str(source), "-B", str(build)], check=True,
                   capture_output=True, text=True)
    replay = [x for x in (build / "replay.txt").read_text().split(";") if x]
    sdk = [x for x in (build / "sdk.txt").read_text().split(";") if x and x != "_sdk-NOTFOUND"]
    assert sdk == replay == expected
    module = _module(tmp_path / "native", compile_options=tuple(replay))
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(toolchain, "_pops_module", lambda: module)
    monkeypatch.setattr(toolchain.sys, "platform", "linux")
    flags, _ = toolchain._native_kokkos_flags()
    for flag in expected:
        if flag.startswith("SHELL:"):
            assert ["-Xcudafe", "--diag_error=20013"] == flags[-2:]
        else:
            assert flag in flags
    if expected:
        module.__kokkos_contract__["compile_options"] = tuple(expected[:-1])
        with pytest.raises(RuntimeError, match="does not authenticate"):
            toolchain._native_kokkos_flags()


@pytest.mark.parametrize("payload", [
    "SHELL:-Xcudafe --diag_error=20013",
    " quote\"\\ slash;pipe|line\nnext\tUTF8:é ",
    "",
])
def test_native_contract_header_preserves_compiled_bytes(tmp_path, payload):
    """The production CMake serializer survives the actual C++ preprocessor."""
    root = Path(__file__).resolve().parents[4]
    helper = root / "cmake/PopsNativeContractHeader.cmake"
    header = tmp_path / "contract.hpp"
    script = tmp_path / "contract.cmake"
    script.write_text(
        f'include("{helper}")\n'
        f'file(WRITE "{header}" "#pragma once\n")\n'
        f'_pops_append_native_contract_string("{header}" PROBE [==[{payload}]==])\n'
    )
    subprocess.run([shutil.which("cmake"), "-P", str(script)], check=True,
                   capture_output=True, text=True)
    source = tmp_path / "main.cpp"
    source.write_text(
        '#include "contract.hpp"\n#include <cstdio>\n'
        'int main() { return std::fwrite(PROBE, 1, sizeof(PROBE)-1, stdout) '
        '== sizeof(PROBE)-1 ? 0 : 1; }\n'
    )
    compiler = shutil.which("c++")
    assert compiler is not None
    executable = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++20", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True)
    assert result.stdout == payload.encode("utf-8")


def test_native_kokkos_contract_uses_header_for_every_string(tmp_path):
    """Execute the production binding mapping, including whitespace-bearing options."""
    root = Path(__file__).resolve().parents[4]
    production = (root / "python/CMakeLists.txt").read_text()
    begin = production.index('include("${CMAKE_SOURCE_DIR}/cmake/PopsNativeContractHeader.cmake")')
    end = production.index("\n\n# Contrat MPI::MPI_CXX", begin)
    block = production[begin:end]
    assert "target_compile_definitions" not in block
    fields = dict(re.findall(
        r'"\$\{_pops_native_contract_header\}" (POPS_[A-Z_]+) "\$\{([^}]+)\}"', block))
    assert len(fields) == 9
    values = {name: ("SHELL:-Xcudafe --diag_error=20013" if name == "POPS_KOKKOS_COMPILE_OPTIONS"
                     else "20" if name == "POPS_KOKKOS_STANDARD" else "field "+name)
              for name in fields}
    source = tmp_path / "source"
    (source / "cmake").mkdir(parents=True)
    shutil.copyfile(root / "cmake/PopsNativeContractHeader.cmake",
                    source / "cmake/PopsNativeContractHeader.cmake")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\nproject(contract LANGUAGES NONE)\n"
        "add_library(_pops INTERFACE)\n"
        + "".join(f"set({variable} [==[{values[name]}]==])\n"
                  for name, variable in fields.items())
        + block.replace("target_include_directories(_pops PRIVATE",
                        "target_include_directories(_pops INTERFACE") + "\n")
    build = tmp_path / "build"
    subprocess.run([shutil.which("cmake"), "-S", str(source), "-B", str(build)],
                   check=True, capture_output=True, text=True)
    main = tmp_path / "main.cpp"
    main.write_text('#include "pops_native_contract_strings.hpp"\n#include <cstdio>\nint main() {\n'
                    + "".join(f'std::fwrite({name}, 1, sizeof({name})-1, stdout); '
                              'std::fputc(0, stdout);\n' for name in fields)
                    + "}\n")
    compiler = shutil.which("c++")
    assert compiler is not None
    executable = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++20", str(main), "-I", str(build / "native-contract"),
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    actual = subprocess.run([str(executable)], check=True, capture_output=True).stdout.split(b"\0")
    assert actual == [value.encode() for value in values.values()] + [b""]
