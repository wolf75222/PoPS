"""Compile the isolated prepared Cartesian path witness against installed Kokkos.

This source-header test does not load or qualify the generated PoPS runtime.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.python.support.requirements import REPO_ROOT, default_cxx, require_native_or_skip


def _kokkos_config():
    direct = os.environ.get("Kokkos_DIR")
    if direct and (Path(direct) / "KokkosConfig.cmake").is_file():
        return Path(direct)
    prefixes = [
        os.environ.get("POPS_KOKKOS_ROOT"),
        os.environ.get("Kokkos_ROOT"),
        os.environ.get("KOKKOS_ROOT"),
        sys.prefix,
    ]
    for prefix in filter(None, prefixes):
        for suffix in ("lib/cmake/Kokkos", "lib64/cmake/Kokkos", "share/cmake/Kokkos"):
            candidate = Path(prefix) / suffix
            if (candidate / "KokkosConfig.cmake").is_file():
                return candidate
    require_native_or_skip("no installed Kokkos CMake package", optional_skip=pytest.skip)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.compiler
def test_prepared_cartesian_path_transactions_and_exact_gaussian_balances(tmp_path):
    cmake, compiler = shutil.which("cmake"), default_cxx()
    if not cmake or not compiler:
        require_native_or_skip("CMake and a C++ compiler are required", optional_skip=pytest.skip)
    kokkos = _kokkos_config()
    retained = os.environ.get("POPS_FANLI15_CARTESIAN_EVIDENCE")
    directory = Path(retained) if retained else tmp_path / "cartesian-path"
    directory.mkdir(parents=True, exist_ok=False)
    sources = [
        "tests/cpp/support/fan_li15_cartesian_witness.cpp",
        "include/pops/numerics/fv/fan_li15_path_flux.hpp",
        "include/pops/numerics/fv/flux_interfaces.hpp",
        "include/pops/numerics/fv/numerical_flux.hpp",
        "include/pops/numerics/spatial/operators/cartesian_operator.hpp",
        "include/pops/physics/composition/composite.hpp",
        "include/pops/numerics/moments/fan_li15_path.hpp",
        "include/pops/numerics/moments/fan_li15_interface.hpp",
    ]
    hashes = {name: _hash(REPO_ROOT / name) for name in sources}
    for name in sources:
        target = directory / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / name).read_bytes())
    (directory / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.20)
project(fan_li15_cartesian_witness LANGUAGES CXX)
find_package(Kokkos REQUIRED CONFIG)
add_executable(fan_li15_cartesian_witness "${POPS_SOURCE}/tests/cpp/support/fan_li15_cartesian_witness.cpp")
target_compile_features(fan_li15_cartesian_witness PRIVATE cxx_std_20)
target_compile_definitions(fan_li15_cartesian_witness PRIVATE POPS_NATIVE_DIM=2 POPS_HAS_KOKKOS=1)
target_include_directories(fan_li15_cartesian_witness PRIVATE "${POPS_SOURCE}/include")
target_link_libraries(fan_li15_cartesian_witness PRIVATE Kokkos::kokkos)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(fan_li15_cartesian_witness PRIVATE -O2 -fno-fast-math -ffp-contract=off)
else()
  message(FATAL_ERROR "This numerical certificate witness requires a reviewed strict-FP compiler contract")
endif()
"""
    )
    build = directory / "build"
    commands = [
        [
            cmake,
            "-S",
            str(directory),
            "-B",
            str(build),
            f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DKokkos_DIR={kokkos}",
            f"-DCMAKE_PREFIX_PATH={kokkos.parents[2]}",
            f"-DPOPS_SOURCE={REPO_ROOT}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        [cmake, "--build", str(build), "--parallel", "2"],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (directory / f"build-{index}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
    binary = build / "fan_li15_cartesian_witness"
    runs = []
    for workers in (1, 2):
        environment = {**os.environ, "OMP_NUM_THREADS": str(workers), "OMP_PROC_BIND": "false"}
        result = subprocess.run(
            [str(binary)], env=environment, capture_output=True, text=True, timeout=90
        )
        (directory / f"omp{workers}.stdout").write_text(result.stdout)
        (directory / f"omp{workers}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert len(report["tests"]) == 20
        assert report["repeat_evaluation_kokkos_allocations"] == 0
        if report["execution_space"] == "OpenMP":
            assert report["concurrency"] == workers
        runs.append(report)
    for key in ("speed", "shared_F40", "tests"):
        assert runs[0][key] == runs[1][key]
    assert hashes == {name: _hash(REPO_ROOT / name) for name in sources}, (
        "source changed during witness"
    )
    (directory / "receipt.json").write_text(
        json.dumps(
            {
                "scope": "isolated Cartesian Kokkos source-header witness; no generated runtime/AMR/MPI/time evolution",
                "commands": commands,
                "source_sha256": hashes,
                "binary_sha256": _hash(binary),
                "kokkos_config": str(kokkos),
                "kokkos_config_sha256": _hash(kokkos / "KokkosConfig.cmake"),
                "runs": runs,
            },
            indent=2,
        )
        + "\n"
    )
