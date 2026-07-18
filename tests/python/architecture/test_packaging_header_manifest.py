"""Installed C++ headers and wheel inputs are exact, categorized and tracked-only."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


packaging = _load("scripts/check_packaging_manifest.py", "_pops_packaging_manifest_test")
toolchain = _load("python/pops/codegen/toolchain.py", "_pops_toolchain_manifest_test")


def test_manifest_exactly_classifies_all_tracked_headers_and_include_fragments():
    manifest = packaging.read_manifest(ROOT)
    tracked = packaging.git_tracked_files(ROOT)
    tracked_headers = {
        path.relative_to("include")
        for path in tracked
        if path.parts[:2] == ("include", "pops")
        and path.suffix in packaging.HEADER_SUFFIXES
    }
    categories = (
        manifest.api,
        manifest.abi,
        manifest.sdk_root,
        manifest.sdk_support,
        manifest.test_only,
    )

    assert manifest.all_headers == tracked_headers
    assert all(categories)
    assert sum(map(len, categories)) == len(manifest.all_headers)
    assert set(manifest.sdk_support).isdisjoint(manifest.standalone_headers)
    assert set(manifest.installed_headers) == (
        set(manifest.api)
        | set(manifest.abi)
        | set(manifest.sdk_root)
        | set(manifest.sdk_support)
    )
    assert PurePosixPath(
        "pops/runtime/config/generated_route_accessors.inc"
    ) in manifest.sdk_support


def test_cmake_source_install_wheel_and_signature_use_the_shared_contract():
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    source_cmake = (ROOT / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
    python_cmake = (ROOT / "python" / "CMakeLists.txt").read_text(encoding="utf-8")
    cmake_contract = (ROOT / "cmake" / "PopsPublicHeaders.cmake").read_text(encoding="utf-8")

    assert 'include(cmake/PopsPublicHeaders.cmake)' in root_cmake
    assert 'pops_install_headers("${CMAKE_INSTALL_INCLUDEDIR}")' in root_cmake
    assert 'pops_install_headers("pops/include")' in python_cmake
    assert "POPS_INSTALLED_HEADERS" in cmake_contract
    assert "POPS_INSTALLED_HEADER_ROWS" in cmake_contract
    assert "pops_compute_header_signature(POPS_NATIVE_HEADER_SIGNATURE)" in source_cmake
    assert "${_pops_header_category} ${_pops_header}" in cmake_contract
    assert "install(DIRECTORY include/pops" not in root_cmake
    assert 'install(DIRECTORY "${_pops_include}/pops"' not in python_cmake
    assert "POPS_PUBLIC_HEADERS" not in source_cmake + cmake_contract
    assert "pops_install_public_headers" not in root_cmake + python_cmake + cmake_contract


def _fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "include" / "pops").mkdir(parents=True)
    (tmp_path / "python" / "pops").mkdir(parents=True)
    (tmp_path / "include" / "pops_headers.manifest").write_text(
        "\n".join(
            (
                "api pops/api.hpp",
                "api pops/api_helper.hpp",
                "abi pops/abi.h",
                "sdk-root pops/generated_root.hpp",
                "sdk-support pops/generated_support.inc",
                "test-only pops/test_support.hpp",
                "",
            )
        ),
        encoding="utf-8",
    )
    return tmp_path


def _fixture_headers() -> set[PurePosixPath]:
    return {
        PurePosixPath("include/pops/api.hpp"),
        PurePosixPath("include/pops/api_helper.hpp"),
        PurePosixPath("include/pops/abi.h"),
        PurePosixPath("include/pops/generated_root.hpp"),
        PurePosixPath("include/pops/generated_support.inc"),
        PurePosixPath("include/pops/test_support.hpp"),
    }


def test_tracked_only_preflight_rejects_duplicate_contaminants(tmp_path):
    root = _fixture_root(tmp_path)
    tracked = _fixture_headers() | {PurePosixPath("python/pops/__init__.py")}
    physical = tracked | {
        PurePosixPath("include/pops/api 2.hpp"),
        PurePosixPath("python/pops/module 3.py"),
    }
    with pytest.raises(packaging.PackagingManifestError, match="untracked packaging inputs") as err:
        packaging.validate_packaging_inputs(root, tracked=tracked, physical=physical)
    assert "api 2.hpp" in str(err.value)
    assert "module 3.py" in str(err.value)


def test_tracked_only_preflight_accepts_an_exact_snapshot(tmp_path):
    root = _fixture_root(tmp_path)
    tracked = _fixture_headers() | {
        PurePosixPath("python/pops/__init__.py"),
        PurePosixPath("python/pops/_pops.pyi"),
        PurePosixPath("python/pops/py.typed"),
    }
    manifest = packaging.validate_packaging_inputs(root, tracked=tracked, physical=tracked)
    assert tuple(map(str, manifest.sdk_support)) == ("pops/generated_support.inc",)
    assert tuple(map(str, manifest.test_only)) == ("pops/test_support.hpp",)


def test_tracked_only_preflight_rejects_a_header_outside_the_manifest(tmp_path):
    root = _fixture_root(tmp_path)
    tracked = _fixture_headers() | {PurePosixPath("include/pops/new.hpp")}
    with pytest.raises(packaging.PackagingManifestError, match="tracked headers outside"):
        packaging.validate_packaging_inputs(root, tracked=tracked, physical=tracked)


def _write_fixture_contents(include: Path) -> None:
    contents = {
        "api.hpp": "api-v1",
        "api_helper.hpp": "api-helper-v1",
        "abi.h": "abi-v1",
        "generated_root.hpp": "sdk-root-v1",
        "generated_support.inc": "sdk-support-v1",
        "test_support.hpp": "test-v1",
    }
    for name, content in contents.items():
        (include / "pops" / name).write_text(content, encoding="utf-8")


def test_header_signature_authenticates_categories_paths_and_all_installed_bytes(tmp_path):
    include = _fixture_root(tmp_path) / "include"
    _write_fixture_contents(include)
    (include / "pops" / "api 2.hpp").write_text("duplicate-v1", encoding="utf-8")
    baseline = toolchain.pops_header_signature(include)

    for name in ("api.hpp", "abi.h", "generated_root.hpp", "generated_support.inc"):
        path = include / "pops" / name
        original = path.read_text(encoding="utf-8")
        path.write_text(original + "-changed", encoding="utf-8")
        assert toolchain.pops_header_signature(include) != baseline
        path.write_text(original, encoding="utf-8")

    (include / "pops" / "test_support.hpp").write_text("test-v2", encoding="utf-8")
    (include / "pops" / "api 2.hpp").write_text("duplicate-v2", encoding="utf-8")
    assert toolchain.pops_header_signature(include) == baseline

    manifest_path = include / "pops_headers.manifest"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace("api pops/api.hpp", "abi pops/api.hpp"),
        encoding="utf-8",
    )
    assert toolchain.pops_header_signature(include) != baseline


def test_cmake_and_python_compute_the_same_exact_header_signature(tmp_path):
    output = tmp_path / "signature.txt"
    script = tmp_path / "signature.cmake"
    script.write_text(
        f'include("{(ROOT / "cmake" / "PopsPublicHeaders.cmake").as_posix()}")\n'
        "pops_compute_header_signature(signature)\n"
        f'file(WRITE "{output.as_posix()}" "${{signature}}")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == toolchain.pops_header_signature(ROOT / "include")


def _kokkos_include() -> Path | None:
    roots = [
        os.environ.get("POPS_KOKKOS_ROOT"),
        os.environ.get("Kokkos_ROOT"),
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
    ]
    for raw in roots:
        if not raw:
            continue
        root = Path(raw)
        candidates = (root, root / "include")
        for candidate in candidates:
            if (candidate / "Kokkos_Core.hpp").is_file():
                return candidate.resolve()
    return None


def _compile_staged_root(
    *,
    compiler: list[str],
    wheel_include: Path,
    kokkos_include: Path,
    temporary: Path,
    name: str,
    roots: tuple[str, ...],
) -> None:
    source = temporary / f"{name}.cpp"
    depfile = temporary / f"{name}.d"
    source.write_text(
        "".join(f"#include <{header}>\n" for header in roots)
        + "int main() { return 0; }\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for variable in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "POPS_INCLUDE"):
        environment.pop(variable, None)
    command = [
        *compiler,
        "-std=c++20",
        "-fsyntax-only",
        "-DPOPS_HAS_KOKKOS",
        # A configured Kokkos-OpenMP header checks _OPENMP even for syntax-only compilation.
        "-D_OPENMP=201511",
        "-I",
        str(wheel_include),
        "-isystem",
        str(kokkos_include),
        "-MMD",
        "-MF",
        str(depfile),
        str(source),
    ]
    result = subprocess.run(
        command,
        cwd=temporary,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"{name} staged generated-loader root did not compile:\n"
        f"command: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
    )
    dependencies = depfile.read_text(encoding="utf-8", errors="replace")
    assert str((ROOT / "include").resolve()) not in dependencies


def test_wheel_style_staged_headers_compile_system_and_amr_generated_roots(tmp_path):
    """A generated module compiles using only wheel-owned PoPS headers, never the checkout."""
    compiler = shlex.split(os.environ.get("CXX", "")) or [shutil.which("c++") or ""]
    kokkos_include = _kokkos_include()
    if not compiler[0] or kokkos_include is None:
        pytest.skip("a C++ compiler and an installed Kokkos include tree are required")

    manifest = packaging.validate_packaging_inputs(ROOT)
    project = tmp_path / "header-stage-project"
    build = tmp_path / "header-stage-build"
    wheel = tmp_path / "wheel"
    project.mkdir()
    project.joinpath("CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\n"
        "project(pops_header_stage LANGUAGES NONE)\n"
        f'include("{(ROOT / "cmake" / "PopsPublicHeaders.cmake").as_posix()}")\n'
        'pops_install_headers("pops/include")\n',
        encoding="utf-8",
    )
    for command in (
        ["cmake", "-S", str(project), "-B", str(build), f"-DCMAKE_INSTALL_PREFIX={wheel}"],
        ["cmake", "--install", str(build)],
    ):
        result = subprocess.run(
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    wheel_include = wheel / "pops" / "include"
    staged_headers = {
        PurePosixPath(path.relative_to(wheel_include).as_posix())
        for path in wheel_include.rglob("*")
        if path.is_file() and path.suffix in packaging.HEADER_SUFFIXES
    }
    assert staged_headers == set(manifest.installed_headers)
    assert (wheel_include / packaging.MANIFEST_REL.name).is_file()

    assert toolchain.pops_header_signature(wheel_include) == toolchain.pops_header_signature(
        ROOT / "include"
    )
    assert not any((wheel_include / path).exists() for path in manifest.test_only)

    _compile_staged_root(
        compiler=compiler,
        wheel_include=wheel_include,
        kokkos_include=kokkos_include,
        temporary=tmp_path,
        name="system_generated_loader",
        roots=(
            "pops/runtime/builders/compiled/dsl_block.hpp",
            "pops/runtime/program/program_context.hpp",
        ),
    )
    _compile_staged_root(
        compiler=compiler,
        wheel_include=wheel_include,
        kokkos_include=kokkos_include,
        temporary=tmp_path,
        name="amr_generated_loader",
        roots=(
            "pops/runtime/builders/compiled/amr_dsl_block.hpp",
            "pops/runtime/program/amr_program_context.hpp",
        ),
    )
    _compile_staged_root(
        compiler=compiler,
        wheel_include=wheel_include,
        kokkos_include=kokkos_include,
        temporary=tmp_path,
        name="field_nullspace_workspace_public_api",
        roots=(
            "pops/numerics/elliptic/interface/field_nullspace_workspace.hpp",
        ),
    )
