"""No JIT: adversarial metadata contracts for the versioned performance tooling."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT / "docs/development/api_040"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DOC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("joint_reconstruction_benchmark_v1_2")
R = load("joint_reconstruction_resource_probe_v1_2")
NATIVE = ("compiler=Apple LLVM 21.0.0;std=202002L;headers=abc;kokkos=1;"
          "stdlib=libc++_210106;mpi=1;mpi_abi=mpich-hash;dim=2")


def artifact(**changes):
    return SimpleNamespace(**({"abi_key": "abc|clang++|c++20|dim=2",
                              "cxx": "clang++", "std": "c++20"} | changes))


def test_distinct_abi_representations_and_exact_metadata():
    row = B._artifact_metadata(artifact(), NATIVE, "abc")
    assert row["abi_key"] == "abc|clang++|c++20|dim=2"
    assert row["native_abi_key"] == NATIVE
    assert row["abi_environment"]["mpi_abi"] == "mpich-hash"
    assert "headers" not in row["abi_environment"]
    with pytest.raises(ValueError):
        B._abi_environment(row["abi_key"])


@pytest.mark.parametrize("key", [
    NATIVE + ";dim=2", NATIVE.replace("mpi=1;", ""),
    NATIVE.replace("mpi_abi=mpich-hash", "mpi_abi="),
    NATIVE.replace("dim=2", "dim=3"), NATIVE.replace("mpi=1", "mpi=0"),
])
def test_native_abi_fail_closed(key):
    with pytest.raises(ValueError):
        B._artifact_metadata(artifact(), key, "abc")


@pytest.mark.parametrize("changes", [
    {"abi_key": "other|clang++|c++20|dim=2"}, {"cxx": "other-cxx"},
    {"abi_key": "abc|clang++|c++20|dim=2|dim=2"},
    {"abi_key": "abc|clang++|c++20|dim=3"},
    {"abi_key": "abc|clang++|c++17|dim=2", "std": "c++17"},
])
def test_artifact_sdk_compiler_dimension_and_standard_must_agree(changes):
    with pytest.raises(ValueError):
        B._artifact_metadata(artifact(**changes), NATIVE, "abc")


def test_only_header_identity_may_differ_across_native_releases():
    assert B._abi_environment(NATIVE) == B._abi_environment(NATIVE.replace("headers=abc", "headers=def"))
    for old, new in (("mpich-hash", "other-mpi"), ("LLVM 21", "LLVM 22"),
                     ("libc++_210106", "libstdc++"), ("kokkos=1", "kokkos=0")):
        assert B._abi_environment(NATIVE) != B._abi_environment(NATIVE.replace(old, new))


def test_companion_pins_exact_v12_and_keeps_missing_counters_unavailable():
    assert R.V1_2_SHA256 == hashlib.sha256((DOC / "joint_reconstruction_benchmark_v1_2.py").read_bytes()).hexdigest()
    assert R._frozen_v1_2().SCHEMA == B.SCHEMA
    assert R._selected_counters({})["scratch_peak_bytes"]["available"] is False
    assert R._COUNTER_UNITS["scratch_peak_bytes"] == "bytes_largest_single_scratch_buffer"


def test_metadata_precedes_any_cold_jit_on_both_drivers():
    for name in ("joint_reconstruction_benchmark_v1_2", "joint_reconstruction_resource_probe_v1_2"):
        tree = ast.parse((DOC / (name + ".py")).read_text())
        driver = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_driver")
        text = ast.unparse(driver)
        assert text.index("'metadata'") < text.index("cold =" if "benchmark" in name else "rows =")
    worker = ast.unparse(next(node for node in ast.parse((DOC / "joint_reconstruction_benchmark_v1_2.py").read_text()).body
                            if isinstance(node, ast.FunctionDef) and node.name == "_worker"))
    assert worker.index("args.phase == 'metadata'") < worker.index("pops.compile(")


def test_dso_paths_are_mapping_property_not_callable(tmp_path):
    paths = [tmp_path / name for name in ("model.so", "program.so", "layout.so")]
    for path in paths:
        path.write_bytes(b"authentic-test-content")
    obj = SimpleNamespace(blocks=[SimpleNamespace(model=SimpleNamespace(so_path=str(paths[0])))],
                          so_path=str(paths[1]), layout_program_paths={"layout": str(paths[2])})
    assert len(B._dso_sizes(obj)) == 3


def test_private_loader_copy_excluded_only_by_authenticated_content(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    original = tmp_path / "compiled.so"
    original.write_bytes(b"exact authenticated DSO")
    copy = tmp_path / "pops-native-random" / "image-random.dylib"
    copy.parent.mkdir()
    copy.write_bytes(original.read_bytes())
    unknown = copy.with_name("image-unknown.dylib")
    unknown.write_bytes(b"different dependency")
    owned = {str(original): {"sha256": B._sha(original)}}
    result = B._external_image_fingerprint(root, owned, [original, copy, unknown])
    assert result["images"] == {str(unknown): B._sha(unknown)}
    assert result["recognized_owned_copies"] == {str(copy): B._sha(original)}
    copy.write_bytes(b"tampered private image")
    assert str(copy) in B._external_image_fingerprint(root, owned, [copy])["images"]


def test_case_and_measurement_boundaries_are_identical_to_frozen_v1():
    old = ast.parse((DOC / "joint_reconstruction_benchmark.py").read_text())
    new = ast.parse((DOC / "joint_reconstruction_benchmark_v1_2.py").read_text())
    for name in ("_case", "_initial", "_execution_context", "_run_once"):
        before = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
        after = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(before) == ast.dump(after), name
    baseline = load("joint_reconstruction_benchmark")
    for name in ("N", "WIDTH", "STEPS", "DT", "WARMUPS", "SAMPLES", "RTOL", "ATOL",
                 "MASS_ATOL", "COMPILE_TIMEOUT_S", "RUNTIME_TIMEOUT_S", "ORDER"):
        assert getattr(baseline, name) == getattr(B, name), name
