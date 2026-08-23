"""Architecture contract for the closed native-variant manifest."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRITER = ROOT / "scripts" / "write_native_variant_manifest.py"
PROVER = ROOT / "scripts" / "prove_installed_wheel.py"
PYTHON_CMAKE = ROOT / "python" / "CMakeLists.txt"
TEST_MANIFEST = ROOT / "tests" / "test_manifest.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYTEST_CONFTEST = ROOT / "tests" / "python" / "conftest.py"
SERIAL_PARENT_MPI_SMOKE = ROOT / "tests" / "cmake" / "run_serial_parent_mpi_target.cmake"
HDF5_WITHOUT_MPI_SMOKE = ROOT / "tests" / "cmake" / "expect_hdf5_without_mpi_rejected.cmake"
BUILD_FINGERPRINT = "a" * 64


def _writer():
    spec = importlib.util.spec_from_file_location("_native_variant_manifest_test", WRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prover():
    spec = importlib.util.spec_from_file_location("_native_variant_wheel_proof_test", PROVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extension(native_root: Path, dimension: int, payload: bytes = b"native") -> Path:
    path = native_root / f"dim{dimension}" / (
        "_pops" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _module(extension: Path, dimension: int) -> ModuleType:
    module = ModuleType("pops._pops")
    module.__file__ = str(extension)
    module.__native_dimension__ = dimension
    module.__version__ = "1.2.3"
    module.__has_mpi__ = dimension == 3
    module.__has_kokkos__ = True
    module.__build_fingerprint__ = BUILD_FINGERPRINT
    module.abi_key = lambda: f"abi-dim{dimension}"
    return module


def test_writer_extracts_compiled_facts_and_atomically_merges_dimensions(tmp_path, monkeypatch):
    writer = _writer()
    native_root = tmp_path / "pops" / "_native"
    manifest = native_root / "variants.json"
    dim1 = _extension(native_root, 1, b"dim1")
    dim3 = _extension(native_root, 3, b"dim3")
    modules = {dim1.resolve(): _module(dim1, 1), dim3.resolve(): _module(dim3, 3)}
    monkeypatch.setattr(writer, "_load_exact_extension", lambda path: modules[path])

    writer.update_manifest(manifest, dim3, dimension=3, version="1.2.3")
    rows = writer.update_manifest(manifest, dim1, dimension=1, version="1.2.3")

    assert [row["dimension"] for row in rows] == [1, 3]
    assert rows[0] == {
        "dimension": 1,
        "path": f"dim1/{dim1.name}",
        "sha256": hashlib.sha256(b"dim1").hexdigest(),
        "version": "1.2.3",
        "abi_key": "abi-dim1",
        "build_fingerprint": BUILD_FINGERPRINT,
        "has_mpi": False,
        "has_kokkos": True,
    }
    assert not tuple(native_root.glob(".variants.*"))


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../dim2/_pops.so", "canonical and relative"),
        ("dim1/_pops.so", "for its dimension"),
        ("dim2/not_pops.so", "EXT_SUFFIX"),
    ],
)
def test_manifest_rejects_escaping_or_mislabeled_paths(path, message):
    writer = _writer()
    payload = {
        "schema_version": 2,
        "variants": [{
            "dimension": 2,
            "path": path,
            "sha256": "a" * 64,
            "version": "1.0.0",
            "abi_key": "abi",
            "build_fingerprint": BUILD_FINGERPRINT,
            "has_mpi": False,
            "has_kokkos": True,
        }],
    }

    with pytest.raises(writer.NativeVariantManifestError, match=message):
        writer.validate_manifest_payload(payload)


def test_manifest_requires_unique_sorted_rows_and_exact_expected_set():
    writer = _writer()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    def row(dimension):
        return {
            "dimension": dimension,
            "path": f"dim{dimension}/_pops{suffix}",
            "sha256": f"{dimension}" * 64,
            "version": "1.0.0",
            "abi_key": f"abi-{dimension}",
            "build_fingerprint": BUILD_FINGERPRINT,
            "has_mpi": False,
            "has_kokkos": True,
        }

    with pytest.raises(writer.NativeVariantManifestError, match="unique and sorted"):
        writer.validate_manifest_payload(
            {"schema_version": 2, "variants": [row(3), row(1)]}
        )
    with pytest.raises(writer.NativeVariantManifestError, match="explicit set"):
        writer.validate_manifest_payload(
            {"schema_version": 2, "variants": [row(1), row(3)]},
            expected_dimensions=(1, 2, 3),
        )


def test_writer_cli_is_fully_explicit():
    source = WRITER.read_text(encoding="utf-8")

    for option in ("--extension", "--manifest", "--dimension", "--version"):
        assert option in source
    assert "required=True" in source
    assert 'name = "pops._pops"' in source


@pytest.mark.parametrize("fingerprint", ("A" * 64, "a" * 63, "fingerprint"))
def test_manifest_rejects_malformed_build_fingerprint(fingerprint):
    writer = _writer()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    payload = {
        "schema_version": 2,
        "variants": [{
            "dimension": 2,
            "path": f"dim2/_pops{suffix}",
            "sha256": "a" * 64,
            "version": "1.0.0",
            "abi_key": "abi",
            "build_fingerprint": fingerprint,
            "has_mpi": False,
            "has_kokkos": True,
        }],
    }

    with pytest.raises(writer.NativeVariantManifestError, match="build[_ ]fingerprint"):
        writer.validate_manifest_payload(payload)


def test_manifest_schema_version_must_be_an_exact_integer():
    writer = _writer()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    payload = {
        "schema_version": 2.0,
        "variants": [{
            "dimension": 2,
            "path": f"dim2/_pops{suffix}",
            "sha256": "a" * 64,
            "version": "1.0.0",
            "abi_key": "abi",
            "build_fingerprint": BUILD_FINGERPRINT,
            "has_mpi": False,
            "has_kokkos": True,
        }],
    }

    with pytest.raises(writer.NativeVariantManifestError, match="schema version"):
        writer.validate_manifest_payload(payload)


def test_writer_refuses_a_missing_or_malformed_module_build_fingerprint(tmp_path, monkeypatch):
    writer = _writer()
    native_root = tmp_path / "pops" / "_native"
    extension = _extension(native_root, 2)
    module = _module(extension, 2)
    monkeypatch.setattr(writer, "_load_exact_extension", lambda _path: module)

    del module.__build_fingerprint__
    with pytest.raises(writer.NativeVariantManifestError, match="build fingerprint"):
        writer.native_variant_row(
            extension, manifest=native_root / "variants.json", dimension=2, version="1.2.3"
        )

    module.__build_fingerprint__ = "B" * 64
    with pytest.raises(writer.NativeVariantManifestError, match="build fingerprint"):
        writer.native_variant_row(
            extension, manifest=native_root / "variants.json", dimension=2, version="1.2.3"
        )


def test_cmake_authenticates_and_installs_the_exact_linked_leaf():
    source = PYTHON_CMAKE.read_text(encoding="utf-8")

    assert "add_custom_command(TARGET _pops POST_BUILD" in source
    assert '"$<TARGET_FILE:_pops>"' in source
    assert '--dimension "${POPS_NATIVE_DIM}"' in source
    assert '--version "${PROJECT_VERSION}"' in source
    assert 'install(FILES "${POPS_PY_NATIVE_MANIFEST}" DESTINATION pops/_native)' in source


def test_cmake_configures_and_exports_one_common_native_build_fingerprint():
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    helper = (ROOT / "cmake" / "PopsNativeBuildFingerprint.cmake").read_text(encoding="utf-8")
    python_cmake = PYTHON_CMAKE.read_text(encoding="utf-8")

    assert "include(cmake/PopsNativeBuildFingerprint.cmake)" in root_cmake
    assert "pops_compute_native_build_fingerprint(POPS_NATIVE_BUILD_FINGERPRINT)" in root_cmake
    assert 'POPS_BUILD_FINGERPRINT="${POPS_NATIVE_BUILD_FINGERPRINT}"' in python_cmake
    assert "function(pops_compute_native_build_fingerprint output)" in helper
    assert "CONFIGURE_DEPENDS" in helper
    assert "CMAKE_CONFIGURE_DEPENDS" in helper
    assert "POPS_HEADER_MANIFEST" in helper
    for excluded_per_variant_fact in ("POPS_NATIVE_DIM", "MPI_ABI", "HDF5"):
        assert excluded_per_variant_fact in helper


def test_process_harness_accepts_only_an_authenticated_selected_nested_leaf(tmp_path):
    source_python = tmp_path / "python"
    native_root = source_python / "pops" / "_native"
    selected = _extension(native_root, 3, b"authenticated Dim=3")
    manifest = native_root / "variants.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "variants": [
                    {
                        "dimension": 3,
                        "path": f"dim3/{selected.name}",
                        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
                        "version": "1.2.3",
                        "abi_key": "abi-dim3",
                        "build_fingerprint": BUILD_FINGERPRINT,
                        "has_mpi": False,
                        "has_kokkos": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    root_residue = source_python / "pops" / (
        "_pops" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    root_residue.write_bytes(b"unauthenticated root residue")
    script = """
import importlib.util
import os
from pathlib import Path
import sys

conftest_path = Path(sys.argv[1])
source_python = Path(sys.argv[2])
selected = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("variant_process_conftest", conftest_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.SOURCE_PYTHON = source_python
module._source_python_has_native_variant.cache_clear()

assert module._source_python_has_native_variant(3)
assert not module._source_python_has_native_variant(2)
usable = module._process_pythonpath(str(source_python), native_dimension=3).split(os.pathsep)
assert str(source_python) in usable

selected.write_bytes(b"tampered selected leaf")
module._source_python_has_native_variant.cache_clear()
assert not module._source_python_has_native_variant(3)
refused = module._process_pythonpath(str(source_python), native_dimension=3).split(os.pathsep)
assert str(source_python) not in refused
unselected = module._process_pythonpath(str(source_python), native_dimension=None).split(os.pathsep)
assert str(source_python) not in unselected
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(PYTEST_CONFTEST), str(source_python), str(selected)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert root_residue.is_file()


def test_ci_consumes_only_the_authenticated_dim2_native_variant():
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    conftest = PYTEST_CONFTEST.read_text(encoding="utf-8")

    for retired_root_contract in (
        "find build-kokkos-py/python/pops -maxdepth 1 -name '_pops*.so'",
        "cp build-mpi/python/pops/_pops*.so",
        "from pops import _pops",
    ):
        assert retired_root_contract not in workflow

    for required_contract in (
        "pops-module-dim2-",
        "pops-module-openmp-dim2-",
        "--exclude='_native/***'",
        "scripts/verify_installed_native.py",
        '--expect-dim "$POPS_NATIVE_DIM" --expect-serial',
        "build-mpi/python/pops/_native/variants.json",
        "build-mpi/python-package/pops/_native/variants.json",
        "build-mpi/python-package/pops/_native/dim2/",
        '--expect-dim "$POPS_NATIVE_DIM" --expect-mpi --expect-parallel-hdf5',
        "_pops = select_native_dimension(2)",
        "select_native_dimension(2); import runpy, sys",
    ):
        assert required_contract in workflow

    native_jobs = (
        ("gate-python-prewarm", "gate-python-build"),
        ("gate-python-build", "gate-python"),
        ("gate-python", "gate-python-compile-cache"),
        ("gate-python-compile-cache", "gate-mpi-prewarm"),
        ("gate-mpi-prewarm", "gate"),
        ("mpi", "gate-openmp-prewarm"),
        ("gate-openmp-prewarm", "kokkos-openmp"),
        ("kokkos-openmp", None),
    )
    for job_name, next_job in native_jobs:
        job = workflow.split(f"\n  {job_name}:\n", 1)[1]
        if next_job is not None:
            job = job.split(f"\n  {next_job}:\n", 1)[0]
        assert 'POPS_NATIVE_DIM: "2"' in job, job_name

    assert 'value = environment.get("POPS_NATIVE_DIM")' in conftest
    assert 'select_native_dimension(native_dimension)' in conftest
    assert 'select_native_dimension(int(sys.argv[1]))' in conftest
    assert 'environment.get("POPS_NATIVE_DIM", "2")' not in conftest


def test_ctest_smokes_and_python_suites_preserve_the_exact_native_dimension():
    root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    serial_smoke = SERIAL_PARENT_MPI_SMOKE.read_text(encoding="utf-8")
    hdf5_smoke = HDF5_WITHOUT_MPI_SMOKE.read_text(encoding="utf-8")
    python_cmake = PYTHON_CMAKE.read_text(encoding="utf-8")

    for test_name in (
        "test_serial_add_subdirectory_ignores_parent_mpi",
        "test_hdf5_without_mpi_rejected",
    ):
        test_block = root_cmake.split(f"NAME {test_name}", 1)[1].split("add_test(", 1)[0]
        assert '"-DPOPS_NATIVE_DIM=${POPS_NATIVE_DIM}"' in test_block

    for smoke in (serial_smoke, hdf5_smoke):
        assert (
            "POPS_NATIVE_DIM" in smoke.split("foreach(_required", 1)[1].split("endforeach()", 1)[0]
        )
        assert '"-DPOPS_NATIVE_DIM=${POPS_NATIVE_DIM}"' in smoke

    pytest_environment = python_cmake.split("set(_pops_py_test_env", 1)[1].split(
        "function(pops_add_pytest_suite", 1
    )[0]
    assert '"POPS_NATIVE_DIM=${POPS_NATIVE_DIM}"' in pytest_environment


def test_ctest_python_mpi_projection_matches_the_manifest_and_dim2_contract():
    source = PYTHON_CMAKE.read_text(encoding="utf-8")
    manifest = TEST_MANIFEST.read_text(encoding="utf-8")

    def manifest_suite(name: str) -> str:
        return manifest.split(f'name = "{name}"', 1)[1].split("\n[[python.suite]]", 1)[0]

    def manifest_entrypoints(name: str) -> tuple[tuple[str, int], ...]:
        return tuple(
            (path, int(nproc))
            for path, nproc in re.findall(
                r'\{ path = "([^"]+)", nproc = (\d+) \}', manifest_suite(name)
            )
        )

    def cmake_entrypoints(variable: str) -> tuple[str, ...]:
        block = source.split(f"set({variable}", 1)[1].split(")", 1)[0]
        return tuple(
            re.findall(
                r"^\s*(tests/python/integration/(?:io|mpi)/test_[^\s]+\.py)\s*$",
                block,
                flags=re.MULTILINE,
            )
        )

    manifest_projection = (
        *manifest_entrypoints("pops_python_integration_io"),
        *manifest_entrypoints("pops_python_integration_mpi"),
    )
    cmake_projection = (
        *cmake_entrypoints("_pops_py_io_mpi_entrypoints"),
        *cmake_entrypoints("_pops_py_mpi_entrypoints"),
    )
    assert len(manifest_projection) == 9
    assert all(nproc == 2 for _, nproc in manifest_projection)
    assert cmake_projection == tuple(path for path, _ in manifest_projection)

    orchestrator = re.search(
        r"set\(_pops_py_rank_change_orchestrator\s+([^\s)]+)\)", source
    )
    assert orchestrator is not None
    assert orchestrator.group(1) == (
        "tests/python/integration/mpi/test_amr_rank_change_restart.py"
    )
    assert (
        "if(POPS_NATIVE_DIM STREQUAL \"2\")" in source
        and source.count("if(POPS_NATIVE_DIM STREQUAL \"2\")") == 1
    )
    assert "if(POPS_USE_MPI)" in source
    test_environment = source.split("set(_pops_py_test_env", 1)[1].split(
        "function(pops_add_pytest_suite", 1
    )[0]
    assert (
        '"${CMAKE_CURRENT_BINARY_DIR}${_pops_py_path_sep}${CMAKE_CURRENT_SOURCE_DIR}'
        '${_pops_py_path_sep}${CMAKE_SOURCE_DIR}"' in source
    )
    assert '"${_pops_py_path_sep}$ENV{PYTHONPATH}"' in source
    for required_environment in (
        '"PYTHONPATH=${_pops_py_test_pythonpath}"',
        '"POPS_NATIVE_DIM=${POPS_NATIVE_DIM}"',
        '"POPS_NATIVE_VARIANTS_ROOT=${CMAKE_CURRENT_BINARY_DIR}/pops/_native"',
        '"POPS_INCLUDE=${CMAKE_SOURCE_DIR}/include"',
        '"POPS_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}"',
        '"POPS_TEST_SOURCE_DIR=${CMAKE_SOURCE_DIR}"',
    ):
        assert required_environment in test_environment
    assert (
        "COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}"
        in source
    )
    assert "${Python_EXECUTABLE} ${MPIEXEC_POSTFLAGS} -u -c" in source
    assert (
        "${Python_EXECUTABLE} ${MPIEXEC_POSTFLAGS} -u -c "
        '"${_pops_py_mpi_bootstrap}"' in source
    )
    assert "from pops._native_selector import select_native_dimension" in source
    assert "select_native_dimension(int(os.environ[\\\"POPS_NATIVE_DIM\\\"]))" in source
    assert "runpy.run_path(sys.argv[1], run_name=\\\"__main__\\\")" in source
    assert source.count("pops_add_mpi_pytest_entrypoint(\"") == 1
    assert 'NAME "pops_python_mpi_${_pops_py_mpi_stem}_np2"' in source
    assert "PROCESSORS 2" in source
    assert source.count('ENVIRONMENT "${_pops_py_test_env};POPS_REQUIRE_MPI_TESTS=1"') == 2
    assert 'WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"' in source
    assert "TIMEOUT 900" in source
    assert "LABELS \"integration;python;mpi;np2\"" in source

    assert "${_pops_py_io_ignore_args}" in source
    assert "${_pops_py_mpi_ignore_args}" in source
    assert source.count('"--ignore=${CMAKE_SOURCE_DIR}/${_pops_py_mpi_entrypoint}"') == 2
    assert '"--ignore=${CMAKE_SOURCE_DIR}/${_pops_py_rank_change_orchestrator}"' in source
    assert source.count("pops_python_mpi_test_amr_rank_change_restart_orchestrator") == 2
    assert "NAME pops_python_mpi_test_amr_rank_change_restart_orchestrator" in source
    orchestrator_properties = source.split(
        "set_tests_properties(pops_python_mpi_test_amr_rank_change_restart_orchestrator", 1
    )[1].split("endif()", 1)[0]
    assert "PROCESSORS 2" in orchestrator_properties
    assert "${MPIEXEC_EXECUTABLE}" not in source.split(
        "NAME pops_python_mpi_test_amr_rank_change_restart_orchestrator", 1
    )[1].split("endif()", 1)[0]


def test_wheel_proof_accepts_an_explicit_fat_set_and_rejects_a_hidden_subset(tmp_path):
    prover = _prover()
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    wheel = tmp_path / "pops-1.2.3-cp312-cp312-any.whl"
    distribution = tmp_path / "site-packages"
    package = distribution / "pops" / "__init__.py"
    manifest = distribution / "pops" / "_native" / "variants.json"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    rows = []
    members = {}
    for dimension in (1, 2, 3):
        payload = f"native Dim={dimension}".encode()
        relative = f"dim{dimension}/_pops{suffix}"
        rows.append({
            "dimension": dimension,
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "version": "1.2.3",
            "abi_key": f"abi-{dimension}",
            "build_fingerprint": BUILD_FINGERPRINT,
            "has_mpi": False,
            "has_kokkos": True,
        })
        members["pops/_native/" + relative] = payload
        installed = distribution / "pops" / "_native" / relative
        installed.parent.mkdir(parents=True)
        installed.write_bytes(payload)
    manifest_payload = {"schema_version": 2, "variants": rows}
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    metadata_payload = "Metadata-Version: 2.3\nName: PoPS\nVersion: 1.2.3\n"
    metadata = distribution / "pops-1.2.3.dist-info" / "METADATA"
    metadata.parent.mkdir()
    metadata.write_text(metadata_payload, encoding="utf-8")
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pops/__init__.py", "")
        archive.writestr("pops/_native/variants.json", json.dumps(manifest_payload))
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr("pops-1.2.3.dist-info/METADATA", metadata_payload)
    direct_url = {
        "archive_info": {"hashes": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}},
        "url": wheel.as_uri(),
    }

    proof = prover.build_proof(
        wheel,
        package_file=package,
        native_manifest=manifest,
        distribution_root=distribution,
        python_executable=Path(sys.executable),
        installed_version="1.2.3",
        direct_url=direct_url,
        expected_dimensions=(1, 2, 3),
    )

    assert [row["dimension"] for row in proof["native_variants"]] == [1, 2, 3]
    with pytest.raises(prover.NativeVariantManifestError, match="explicit set"):
        prover.build_proof(
            wheel,
            package_file=package,
            native_manifest=manifest,
            distribution_root=distribution,
            python_executable=Path(sys.executable),
            installed_version="1.2.3",
            direct_url=direct_url,
            expected_dimensions=(2,),
        )
