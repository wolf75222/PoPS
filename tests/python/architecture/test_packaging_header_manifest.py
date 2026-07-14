"""Installed C++ headers and wheel inputs are manifest-driven and tracked-only."""
from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath
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


def test_manifest_exactly_classifies_all_tracked_headers():
    manifest = packaging.read_manifest(ROOT)
    tracked = packaging.git_tracked_files(ROOT)
    tracked_headers = {
        path.relative_to("include")
        for path in tracked
        if path.parts[:2] == ("include", "pops") and path.suffix in {".h", ".hpp"}
    }
    assert manifest.all_headers == tracked_headers
    assert len(manifest.test_only) == 9
    assert not set(manifest.public).intersection(manifest.test_only)


def test_cmake_source_install_wheel_and_signature_use_the_shared_manifest():
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    python_cmake = (ROOT / "python" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'include(cmake/PopsPublicHeaders.cmake)' in root_cmake
    assert 'pops_install_public_headers("${CMAKE_INSTALL_INCLUDEDIR}")' in root_cmake
    assert 'pops_install_public_headers("pops/include")' in python_cmake
    assert "install(DIRECTORY include/pops" not in root_cmake
    assert 'install(DIRECTORY "${_pops_include}/pops"' not in python_cmake
    assert "foreach(_h IN LISTS POPS_PUBLIC_HEADERS)" in python_cmake


def _fixture_root(tmp_path: Path) -> Path:
    (tmp_path / "include" / "pops").mkdir(parents=True)
    (tmp_path / "python" / "pops").mkdir(parents=True)
    (tmp_path / "include" / "pops_public_headers.manifest").write_text(
        "public pops/api.hpp\ntest-only pops/test_support.hpp\n", encoding="utf-8"
    )
    return tmp_path


def test_tracked_only_preflight_rejects_duplicate_contaminants(tmp_path):
    root = _fixture_root(tmp_path)
    tracked = {
        PurePosixPath("include/pops/api.hpp"),
        PurePosixPath("include/pops/test_support.hpp"),
        PurePosixPath("python/pops/__init__.py"),
    }
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
    tracked = {
        PurePosixPath("include/pops/api.hpp"),
        PurePosixPath("include/pops/test_support.hpp"),
        PurePosixPath("python/pops/__init__.py"),
        PurePosixPath("python/pops/_pops.pyi"),
        PurePosixPath("python/pops/py.typed"),
    }
    manifest = packaging.validate_packaging_inputs(root, tracked=tracked, physical=tracked)
    assert tuple(map(str, manifest.test_only)) == ("pops/test_support.hpp",)


def test_tracked_only_preflight_rejects_a_header_outside_the_manifest(tmp_path):
    root = _fixture_root(tmp_path)
    tracked = {
        PurePosixPath("include/pops/api.hpp"),
        PurePosixPath("include/pops/test_support.hpp"),
        PurePosixPath("include/pops/new.hpp"),
    }
    with pytest.raises(packaging.PackagingManifestError, match="tracked headers outside"):
        packaging.validate_packaging_inputs(root, tracked=tracked, physical=tracked)


def test_header_signature_ignores_test_only_and_untracked_headers(tmp_path):
    include = _fixture_root(tmp_path) / "include"
    (include / "pops" / "api.hpp").write_text("public-v1", encoding="utf-8")
    (include / "pops" / "test_support.hpp").write_text("test-v1", encoding="utf-8")
    (include / "pops" / "api 2.hpp").write_text("duplicate-v1", encoding="utf-8")
    baseline = toolchain.pops_header_signature(include)

    (include / "pops" / "test_support.hpp").write_text("test-v2", encoding="utf-8")
    (include / "pops" / "api 2.hpp").write_text("duplicate-v2", encoding="utf-8")
    assert toolchain.pops_header_signature(include) == baseline

    (include / "pops" / "api.hpp").write_text("public-v2", encoding="utf-8")
    assert toolchain.pops_header_signature(include) != baseline
