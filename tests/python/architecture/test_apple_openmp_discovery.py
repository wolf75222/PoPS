"""The selected Kokkos package and its OpenMP target must share one runtime."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("apple", [True, False])
def test_resolved_kokkos_rebinds_an_existing_openmp_target(tmp_path, apple):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the package-discovery regression")
    for name in ("initial", "resolved"):
        prefix = tmp_path / name
        (prefix / "lib/cmake/Kokkos").mkdir(parents=True)
        (prefix / "include").mkdir()
        (prefix / "lib/libomp.dylib").touch()
    project = tmp_path / "CMakeLists.txt"
    expected = "resolved" if apple else "initial"
    excluded = "initial" if apple else "resolved"
    project.write_text(
        f"""cmake_minimum_required(VERSION 3.21)
project(OpenMPDiscovery NONE)
set(APPLE TRUE)
include("{(ROOT / 'cmake/PopsAppleOpenMP.cmake').as_posix()}")
set(Kokkos_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/initial/lib/cmake/Kokkos")
pops_apple_libomp_hints()
add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
  INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp;-I${{LIBOMP_PREFIX}}/include"
  INTERFACE_INCLUDE_DIRECTORIES "${{LIBOMP_PREFIX}}/include"
  INTERFACE_LINK_LIBRARIES "${{OpenMP_omp_LIBRARY}}")
# The package search resolves a different prefix after FindOpenMP creates its target.
set(Kokkos_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/resolved/lib/cmake/Kokkos")
set(APPLE {"TRUE" if apple else "FALSE"})
pops_apple_libomp_hints()
foreach(property IN ITEMS
    INTERFACE_COMPILE_OPTIONS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_LINK_LIBRARIES)
  get_target_property(value OpenMP::OpenMP_CXX "${{property}}")
  if(value MATCHES "/{excluded}/" OR NOT value MATCHES "/{expected}/")
    message(FATAL_ERROR "Stale OpenMP target ${{property}}: ${{value}}")
  endif()
endforeach()
if(NOT OpenMP_omp_LIBRARY STREQUAL "${{CMAKE_CURRENT_SOURCE_DIR}}/{expected}/lib/libomp.dylib")
  message(FATAL_ERROR "Stale OpenMP runtime cache: ${{OpenMP_omp_LIBRARY}}")
endif()
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
