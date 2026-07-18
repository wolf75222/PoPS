"""Compiler-observed dependency closure for prepared native components."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.python.support.requirements import (
    default_cxx,
    repo_include,
    require_native_or_skip,
)


def _compiler() -> str:
    compiler = default_cxx()
    if compiler is None:
        require_native_or_skip("no C++ compiler available", optional_skip=pytest.skip)
        raise AssertionError("unreachable")
    return compiler


@pytest.mark.parametrize("variable", ("CPATH", "CPLUS_INCLUDE_PATH"))
def test_native_compile_cannot_read_ambient_include_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    from pops.codegen.toolchain import _run_compile

    injected = tmp_path / "ambient-injected"
    injected.mkdir()
    (injected / "pops_ambient_injected_header.hpp").write_text(
        "#pragma once\n", encoding="utf-8"
    )
    source = tmp_path / "probe.cpp"
    source.write_text(
        "#include <pops_ambient_injected_header.hpp>\nint probe() { return 0; }\n",
        encoding="utf-8",
    )
    output = tmp_path / "probe.o"
    monkeypatch.setenv(variable, str(injected))

    with pytest.raises(RuntimeError, match="compiling the .so"):
        _run_compile(
            [_compiler(), "-std=c++20", "-c", str(source), "-o", str(output)],
            "ambient include injection probe",
        )
    assert not output.exists()


def test_native_compile_environment_removes_link_and_flag_injection() -> None:
    from pops.codegen.toolchain import native_compile_environment

    source = {
        "PATH": "/explicit/toolchain/bin",
        "SDKROOT": "/explicit/sdk",
        "CPATH": "/ambient/include",
        "CPLUS_INCLUDE_PATH": "/ambient/cxx/include",
        "LIBRARY_PATH": "/ambient/lib",
        "LD_LIBRARY_PATH": "/ambient/runtime",
        "DYLD_INSERT_LIBRARIES": "/ambient/injected.dylib",
        "CPPFLAGS": "-I/ambient/include",
        "CXXFLAGS": "-include ambient.hpp",
        "LDFLAGS": "-L/ambient/lib -lambient",
    }

    cleaned = native_compile_environment(source)

    assert cleaned == {
        "PATH": "/explicit/toolchain/bin",
        "SDKROOT": "/explicit/sdk",
    }


def test_dsl_optflags_accept_only_closed_path_free_codegen_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pops.codegen.cache import _dsl_optflags, _platform_cache_key

    value = (
        "-O2 -DNDEBUG -march=armv8.2-a+simd -mtune=neoverse-v2 "
        "-ffp-contract=fast -fno-math-errno -funroll-loops"
    )
    monkeypatch.setenv("POPS_DSL_OPTFLAGS", value)

    assert _dsl_optflags() == value.split()
    assert _platform_cache_key().endswith("optflags=" + value)


@pytest.mark.parametrize(
    "value",
    (
        "-O3 -isystem /tmp/unowned -include evil.hpp",
        "-O3 -I/tmp/unowned",
        "-O3 --sysroot=/tmp/sdk",
        "-O3 -B/tmp/toolchain",
        "-O3 -F/tmp/frameworks",
        "-O3 -march=native",
        "-O3 -Wp,-I/tmp/unowned",
        "-O3 -Xclang -load -Xclang /tmp/plugin.dylib",
        "-O3 -fplugin=/tmp/plugin.so",
        "-O3 /tmp/evil.o",
        "-O3 @/tmp/evil.rsp",
        "-O3 -Wl,-rpath,/tmp/evil",
        "-O3 -L/tmp/evil -levil",
    ),
)
def test_dsl_optflags_reject_external_build_inputs_before_compilation(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    from pops.codegen.cache import _dsl_optflags

    monkeypatch.setenv("POPS_DSL_OPTFLAGS", value)
    with pytest.raises(ValueError, match="closed path-free.*allowlist"):
        _dsl_optflags()


def test_mmd_omits_a_forced_isystem_header_so_the_flag_allowlist_must_reject_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real compiler behavior that makes free-form optflags unsafe."""
    from pops.codegen.cache import _dsl_optflags
    from pops.codegen.toolchain import _run_compile

    hidden_root = tmp_path / "hidden-system-root"
    hidden_root.mkdir()
    hidden = hidden_root / "forced_hidden.hpp"
    hidden.write_text("#pragma once\n#define POPS_FORCED_HIDDEN 1\n", encoding="utf-8")
    source = tmp_path / "probe.cpp"
    source.write_text(
        "#ifndef POPS_FORCED_HIDDEN\n#error forced include absent\n#endif\nint probe() { return 0; }\n",
        encoding="utf-8",
    )
    output = tmp_path / "probe.o"
    dependency_file = tmp_path / "probe.d"
    _run_compile(
        [
            _compiler(),
            "-std=c++20",
            "-MMD",
            "-MF",
            str(dependency_file),
            "-isystem",
            str(hidden_root),
            "-include",
            hidden.name,
            "-c",
            str(source),
            "-o",
            str(output),
        ],
        "forced system dependency probe",
    )
    assert output.is_file()
    assert str(hidden) not in dependency_file.read_text(encoding="utf-8")

    monkeypatch.setenv(
        "POPS_DSL_OPTFLAGS", "-O3 -isystem %s -include %s" % (hidden_root, hidden.name)
    )
    with pytest.raises(ValueError, match="closed path-free.*allowlist"):
        _dsl_optflags()


def test_compiler_dependency_closure_accepts_std_pops_kokkos_and_component(
    tmp_path: Path,
) -> None:
    from pops.codegen.toolchain import _run_compile
    from pops.native_components import (
        PreparedNativeComponent,
        compiler_include_roots,
        verify_prepared_native_dependencies,
    )

    component_root = tmp_path / "component source"
    component_header = component_root / "vendor" / "prepared_probe.hpp"
    component_header.parent.mkdir(parents=True)
    component_header.write_text(
        """#pragma once
#include <vector>
#include <Kokkos_Core.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
namespace vendor { inline std::vector<int> probe() { return {1}; } }
""",
        encoding="utf-8",
    )
    component = PreparedNativeComponent.header_only(
        "tests.prepared-native.valid-closure",
        include_root=component_root,
        entry_headers=("vendor/prepared_probe.hpp",),
    )
    staged_root = component.stage_verified(tmp_path / "staged component")
    assert staged_root is not None

    kokkos_root = tmp_path / "authenticated-kokkos"
    kokkos_root.mkdir()
    (kokkos_root / "Kokkos_Core.hpp").write_text("#pragma once\n", encoding="utf-8")
    generated = tmp_path / "generated source.cpp"
    generated.write_text(
        "#include <vendor/prepared_probe.hpp>\nint probe() { return vendor::probe()[0]; }\n",
        encoding="utf-8",
    )
    dependency_file = tmp_path / "generated source.d"
    output = tmp_path / "generated source.o"
    _run_compile(
        [
            _compiler(),
            "-std=c++20",
            "-MMD",
            "-MF",
            str(dependency_file),
            "-MT",
            str(output),
            "-I",
            staged_root,
            "-I",
            repo_include(),
            "-I",
            str(kokkos_root),
            "-c",
            str(generated),
            "-o",
            str(output),
        ],
        "prepared dependency closure probe",
    )

    dependencies = verify_prepared_native_dependencies(
        dependency_file,
        generated_source=generated,
        pops_include_root=repo_include(),
        staged_components=((component, staged_root),),
        toolchain_include_roots=compiler_include_roots(["-I", str(kokkos_root)]),
    )

    assert os.path.realpath(generated) in dependencies
    assert os.path.realpath(Path(staged_root) / "vendor" / "prepared_probe.hpp") in dependencies
    assert (
        os.path.realpath(Path(repo_include()) / "pops/mesh/layout/field_distribution.hpp")
        in dependencies
    )
    assert os.path.realpath(kokkos_root / "Kokkos_Core.hpp") in dependencies


def test_compiler_dependency_closure_rejects_external_angle_include(
    tmp_path: Path,
) -> None:
    from pops.codegen.toolchain import _run_compile
    from pops.native_components import (
        PreparedNativeComponent,
        verify_prepared_native_dependencies,
    )

    external_root = tmp_path / "unowned-include"
    external_header = external_root / "vendor" / "untracked.hpp"
    external_header.parent.mkdir(parents=True)
    external_header.write_text("#pragma once\n", encoding="utf-8")
    component_root = tmp_path / "component"
    component_root.mkdir()
    (component_root / "component.hpp").write_text(
        "#pragma once\n#include <vendor/untracked.hpp>\n", encoding="utf-8"
    )
    component = PreparedNativeComponent.header_only(
        "tests.prepared-native.external-angle",
        include_root=component_root,
        entry_headers=("component.hpp",),
    )
    staged_root = component.stage_verified(tmp_path / "staged")
    assert staged_root is not None
    generated = tmp_path / "generated.cpp"
    generated.write_text("#include <component.hpp>\nint probe() { return 0; }\n", encoding="utf-8")
    dependency_file = tmp_path / "generated.d"
    output = tmp_path / "generated.o"
    _run_compile(
        [
            _compiler(),
            "-std=c++20",
            "-MMD",
            "-MF",
            str(dependency_file),
            "-MT",
            str(output),
            "-I",
            staged_root,
            "-I",
            str(external_root),
            "-c",
            str(generated),
            "-o",
            str(output),
        ],
        "external angle dependency probe",
    )
    assert output.is_file()

    with pytest.raises(RuntimeError, match="outside the authenticated"):
        verify_prepared_native_dependencies(
            dependency_file,
            generated_source=generated,
            pops_include_root=repo_include(),
            staged_components=((component, staged_root),),
            toolchain_include_roots=(),
        )


def test_rejected_dependency_closure_publishes_no_binary_or_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pops.codegen import _compile_drivers as drivers
    from pops.codegen import toolchain
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    external = tmp_path / "unowned.hpp"
    external.write_text("#pragma once\n", encoding="utf-8")
    monkeypatch.setenv("POPS_CODEGEN_DIR", str(tmp_path))
    monkeypatch.setattr(
        drivers, "pops_loader_build_flags", lambda cxx=None: ("c++", [], [])
    )
    monkeypatch.setattr(drivers, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, std: std or "c++23")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "TEST-FEATURES")

    def _make_escape(path: str) -> str:
        return path.replace("\\", "\\\\").replace(" ", "\\ ")

    def _compiler_with_unowned_dependency(command: list[str], _where: str) -> None:
        output = command[command.index("-o") + 1]
        dependency_file = command[command.index("-MF") + 1]
        generated = next(item for item in command if item.endswith("problem.cpp"))
        Path(output).write_bytes(b"UNVERIFIED-BINARY")
        Path(dependency_file).write_text(
            "%s: %s %s\n"
            % (_make_escape(output), _make_escape(generated), _make_escape(str(external))),
            encoding="utf-8",
        )

    monkeypatch.setattr(drivers, "_run_compile", _compiler_with_unowned_dependency)
    program, module = _program_fixture("rejected_dependency")

    with pytest.raises(RuntimeError, match="outside the authenticated"):
        drivers.compile_problem(model=module, time=program, force=True, include=INCLUDE)

    assert not tuple(tmp_path.rglob("*.so"))
    assert not tuple(tmp_path.rglob("*.dylib"))
    assert not tuple(tmp_path.rglob("*.dll"))
    assert not tuple(tmp_path.rglob("*.pops-artifact.json"))
