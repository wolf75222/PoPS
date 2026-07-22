"""Source-ownership fence for the compiled System/AMR/output runtime.

``src/CMakeLists.txt`` is the only native source manifest.  Python owns pybind adapters, tests own
executables, and both consume the same central object targets without compiling private runtime
implementation files a second time.
"""

from __future__ import annotations

import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "src" / "runtime"
BINDINGS = ROOT / "python" / "bindings"
ROOT_CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
SRC_CMAKE = (ROOT / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
PYTHON_CMAKE = (ROOT / "python" / "CMakeLists.txt").read_text(encoding="utf-8")
TESTS_CMAKE = (ROOT / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")
PRESETS = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
PACKAGE_CONFIG = (ROOT / "cmake" / "popsConfig.cmake.in").read_text(encoding="utf-8")


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_python_bindings_contains_only_pybind_core_adapters():
    files = [path for path in BINDINGS.rglob("*") if path.is_file()]
    misplaced = [_rel(path) for path in files if path.relative_to(BINDINGS).parts[0] != "core"]
    assert not misplaced, (
        "python/bindings is reserved for actual module/init adapters; native runtime/builders "
        "belong under src/runtime:\n  " + "\n  ".join(misplaced)
    )

    binding_sources = sorted(path for path in files if path.suffix == ".cpp")
    missing = [
        _rel(path)
        for path in binding_sources
        if path.relative_to(ROOT / "python").as_posix() not in PYTHON_CMAKE
    ]
    assert not missing, "pybind adapter source missing from POPS_MODULE_BINDING_SOURCES: " + str(missing)


def test_no_system_method_definition_lives_below_python_bindings():
    # Method references such as ``&System::step`` are legitimate pybind glue. A qualified name at the
    # start of a C++ declaration/body line is an implementation definition and is forbidden here.
    definition = re.compile(
        r"(?m)^\s*(?:[A-Za-z_]\w*(?:::\w+)*(?:<[^;{}]*>)?[\s*&]+)*"
        r"(?:System|AmrSystem)::(?:~?\w+|operator\S*)\s*\("
    )
    offenders = []
    for source in BINDINGS.rglob("*.cpp"):
        if definition.search(source.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(_rel(source))
    assert not offenders, "System/AmrSystem implementation leaked into pybind adapters: " + str(offenders)


def test_src_cmake_is_the_single_runtime_source_authority():
    assert SRC_CMAKE.count("add_library(pops_runtime_core_objects OBJECT") == 1
    assert re.search(
        r"add_library\(pops_runtime_core\s+STATIC\s+"
        r"\$<TARGET_OBJECTS:pops_runtime_core_objects>\s*\)",
        SRC_CMAKE,
    )
    for text in (PYTHON_CMAKE, TESTS_CMAKE):
        assert not re.search(r"add_library\(\s*pops_runtime_core_objects\b", text)

    for target in ("pops_runtime_system", "pops_runtime_amr", "pops_runtime_output"):
        assert SRC_CMAKE.count(f"add_library({target} OBJECT") == 1
        assert not re.search(rf"add_library\(\s*{target}\b", PYTHON_CMAKE)
        assert not re.search(rf"add_library\(\s*{target}\b", TESTS_CMAKE)

    hand_written_sources = sorted(RUNTIME.rglob("*.cpp"))
    missing = []
    for source in hand_written_sources:
        manifest_path = source.relative_to(ROOT / "src").as_posix()
        if SRC_CMAKE.count(manifest_path) != 1:
            missing.append(manifest_path)
    assert not missing, "runtime .cpp must occur exactly once in src/CMakeLists.txt: " + str(missing)

    for consumer, text in (("python", PYTHON_CMAKE), ("tests", TESTS_CMAKE)):
        relisted = sorted(
            source.relative_to(ROOT / "src").as_posix()
            for source in hand_written_sources
            if source.relative_to(ROOT / "src").as_posix() in text
        )
        assert not relisted, f"{consumer}/CMakeLists.txt relists central runtime sources: {relisted}"


def test_python_and_tests_consume_the_central_targets():
    assert re.search(
        r"target_link_libraries\(\s*_pops\s+PRIVATE\s+"
        r"pops_runtime_core_objects\s+pops_runtime_system\s+"
        r"pops_runtime_amr\s+pops_runtime_output\b",
        PYTHON_CMAKE,
    )
    for target in ("pops_runtime_system", "pops_runtime_amr", "pops_runtime_output"):
        assert target in TESTS_CMAKE, f"tests have no consumers for {target}"

    for target in ("pops_runtime_system", "pops_runtime_amr", "pops_runtime_output"):
        assert re.search(
            rf"target_link_libraries\(\s*{target}\s+PUBLIC\s+pops_runtime_core\s*\)",
            SRC_CMAKE,
        ), f"{target} does not carry the shared runtime ABI authority transitively"

    positions = {
        name: re.search(rf"(?m)^\s*add_subdirectory\({name}\)\s*$", ROOT_CMAKE).start()
        for name in ("src", "tests", "python")
    }
    assert positions["src"] < positions["tests"]
    assert positions["src"] < positions["python"]


def test_installed_package_rehydrates_native_backend_dependencies():
    """Exported interface targets must never name dependencies the config did not recreate."""
    assert "POPS_USE_HDF5 AND NOT POPS_USE_MPI" in ROOT_CMAKE
    assert "native collective HDF5 writer and therefore requires" in ROOT_CMAKE
    assert "if(@POPS_HAS_HDF5@ AND NOT CMAKE_C_COMPILER_LOADED)" in PACKAGE_CONFIG
    assert "find_dependency(MPI REQUIRED COMPONENTS C CXX)" in PACKAGE_CONFIG
    assert "find_dependency(MPI REQUIRED COMPONENTS CXX)" in PACKAGE_CONFIG
    assert "set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)" in ROOT_CMAKE
    assert "set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)" in PACKAGE_CONFIG
    assert "find_dependency(HDF5 MODULE REQUIRED COMPONENTS C)" in PACKAGE_CONFIG
    assert "if(NOT HDF5_IS_PARALLEL)" in PACKAGE_CONFIG
    assert "test_installed_package_consumer" in ROOT_CMAKE
    assert "test_hdf5_without_mpi_rejected" in ROOT_CMAKE


def test_central_targets_preserve_consumer_specific_compile_contracts():
    required = (
        "pops_heavy_test_tu=${POPS_HEAVY_TEST_TU_POOL}",
        "JOB_POOL_COMPILE pops_heavy_test_tu",
        "JOB_POOL_COMPILE pops_heavy_module_tu",
        "POSITION_INDEPENDENT_CODE ON",
        "CXX_VISIBILITY_PRESET hidden",
        "VISIBILITY_INLINES_HIDDEN ON",
        "pops_dev_options",
        "_pops_EXPORTS",
        "POPS_HEADER_SIG=\"${POPS_NATIVE_HEADER_SIGNATURE}\"",
        "$<CONFIG:Release,RelWithDebInfo,MinSizeRel>",
        ":-O0>",
    )
    missing = [fact for fact in required if fact not in SRC_CMAKE]
    assert not missing, "central runtime targets lost compile-contract facts: " + str(missing)
    assert SRC_CMAKE.count("JOB_POOL_COMPILE pops_heavy_test_tu") == 1
    assert SRC_CMAKE.count("JOB_POOL_COMPILE pops_heavy_module_tu") == 1
    assert "elseif(POPS_BUILD_PYTHON)" in SRC_CMAKE

    assert "pops_heavy_module_tu=${POPS_HEAVY_MODULE_TU_POOL}" in ROOT_CMAKE
    assert "POPS_HEAVY_MODULE_TU_POOL" not in SRC_CMAKE
    assert re.search(r"set\(POPS_HEAVY_MODULE_TU_POOL\s+1\s+CACHE STRING", ROOT_CMAKE)
    assert re.search(r"set\(POPS_HEAVY_TEST_TU_POOL\s+1\s+CACHE STRING", ROOT_CMAKE)
    assert "POPS_HEAVY_TU_POOL" not in ROOT_CMAKE + SRC_CMAKE

    configure_presets = {preset["name"]: preset for preset in PRESETS["configurePresets"]}
    for name in ("ci-kokkos", "ci-mpi"):
        assert configure_presets[name]["cacheVariables"]["POPS_HEAVY_TEST_TU_POOL"] == "1"
        assert "POPS_HEAVY_MODULE_TU_POOL" not in configure_presets[name]["cacheVariables"]
