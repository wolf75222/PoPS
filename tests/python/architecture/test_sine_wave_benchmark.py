from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import json
import runpy
import sys
import types
from functools import partial
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CASE = REPOSITORY_ROOT / "benchmarks" / "verification" / "advection" / "sine_wave"
SUPPORT = CASE / "_case_support.py"
MATRIX = CASE / "matrix.v1.json"
MATRIX_DRIVER = CASE / "run_matrix.py"
SCHEMA_VERSION = "pops.sine-wave.v3"


def _import_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _support_scope() -> dict[str, object]:
    return runpy.run_path(str(SUPPORT))


def _compute_metrics():
    return partial(
        _support_scope()["compute_metrics"],
        probe_time=0.37,
        base_cfl=0.4,
        refinement_ratio=2,
    )


def _test_execution_provenance(*, backend: str = "OpenMP") -> dict[str, object]:
    return {
        "runtime": {
            "has_kokkos": True,
            "kokkos_backend": backend,
            "kokkos_device": "host",
            "kokkos_shared_space": "HostSpace",
            "field_memory_space": "HostSpace",
            "kokkos_concurrency": 2,
            "mpi_compiled": False,
            "mpi_active": False,
            "mpi_ranks": 1,
            "communicator": None,
        },
        "environment": {
            "OMP_NUM_THREADS": "2",
            "OMP_PROC_BIND": "false",
            "OMP_PLACES": None,
            "KOKKOS_NUM_THREADS": "2",
            "CUDA_VISIBLE_DEVICES": None,
            "ROCR_VISIBLE_DEVICES": None,
        },
        "host": {
            "node": "test-node",
            "system": "test-system",
            "release": "test-release",
            "machine": "test-machine",
        },
    }


def _test_amr_diagnostics() -> dict[str, object]:
    return {
        "interface_mask": {
            "definition": ("active composite leaf sharing a face with an inactive same-level cell"),
            "connectivity": "face",
            "periodic": True,
            "storage": "{snapshot_prefix}interface_mask[_level_L]",
        },
        "regrid_events": {
            "source": "simulation.amr.explain_regrid()",
            "storage": "{snapshot_prefix}regrid_count and topology_epoch",
            "time_semantics": (
                "a counter increase occurred after the previous snapshot and no later "
                "than the labelled snapshot; the internal event time is not claimed"
            ),
        },
    }


def _write_timeline_case(tmp_path: Path, dimension: int) -> tuple[Path, Path]:
    """Create a small authenticated data-only fixture with nine physical frames."""
    resolution = (6,) * dimension
    coordinates = tuple((np.arange(count, dtype=float) + 0.5) / count for count in resolution)
    field_coordinates = []
    for physical_axis, coordinate in enumerate(coordinates):
        shape = [1] * dimension
        shape[dimension - 1 - physical_axis] = coordinate.size
        field_coordinates.append(coordinate.reshape(shape))
    phase = sum(
        wave * coordinate for wave, coordinate in zip((1, 2, 3), field_coordinates, strict=False)
    )
    times = np.linspace(0.0, 1.0, 9)
    velocity = (1.0,) * dimension
    frequency = sum(wave * speed for wave, speed in zip((1, 2, 3), velocity, strict=False))
    payload: dict[str, np.ndarray] = {"timeline_time": times}
    snapshots = []
    for index, time in enumerate(times):
        exact = 1.0 + 0.1 * np.sin(2.0 * np.pi * (phase - frequency * time))
        numeric = 1.0 + 0.1 * (1.0 - 0.02 * time) * np.sin(
            2.0 * np.pi * (phase - frequency * time - 0.01 * time)
        )
        snapshots.append((numeric, exact))
        prefix = "timeline_%04d_" % index
        payload[prefix + "numeric"] = numeric
        payload[prefix + "exact"] = exact
        payload[prefix + "mask"] = np.ones_like(exact, dtype=bool)
        payload[prefix + "interface_mask"] = np.zeros_like(exact, dtype=bool)
        payload[prefix + "patch_boxes"] = np.empty((0,), dtype=np.int64)
        payload[prefix + "regrid_count"] = np.asarray(0, dtype=np.int64)
        payload[prefix + "topology_epoch"] = np.asarray(0, dtype=np.int64)
        for name, coordinate in zip(("x", "y", "z"), coordinates, strict=False):
            payload[prefix + name] = coordinate
    payload["numeric"], payload["exact"] = snapshots[-1]
    payload["mask"] = np.ones_like(payload["exact"], dtype=bool)
    payload["interface_mask"] = np.zeros_like(payload["exact"], dtype=bool)
    payload["patch_boxes"] = np.empty((0,), dtype=np.int64)
    payload["regrid_count"] = np.asarray(0, dtype=np.int64)
    payload["topology_epoch"] = np.asarray(0, dtype=np.int64)
    for name, coordinate in zip(("x", "y", "z"), coordinates, strict=False):
        payload[name] = coordinate
    timeline = {
        "frames": len(times),
        "times": times.tolist(),
        "storage_prefix": "timeline_{index:04d}_",
    }
    wave_numbers = list((1, 2, 3)[:dimension])
    execution = _test_execution_provenance()
    source_fingerprint = "a" * 64
    artifact = {
        "semantic_identity": "test-semantic",
        "artifact_identity": "test-artifact",
        "bind_identity": "test-bind",
    }
    method = {
        "time": "SSPRK2",
        "reconstruction": "MUSCL(VanLeer)",
        "riemann": "ScalarUpwind",
        "cfl": 0.4 / dimension,
        "cfl_base": 0.4,
        "cfl_effective": 0.4 / dimension,
        "cfl_formula": "base_cfl / dimension",
    }
    amr_diagnostics = _test_amr_diagnostics()
    configuration = {
        "case": "periodic_sine_wave_advection",
        "dimension": dimension,
        "resolution": list(resolution),
        "mode": "diagonal",
        "wave_numbers": wave_numbers,
        "velocity": list(velocity),
        "epsilon": 0.1,
        "probe_time": 0.37,
        "period": 1.0,
        "cycles": 1,
        "final_time": 1.0,
        "layout": "uniform",
        "subcycling": "synchronous",
        "block_size": 16,
        "patch_marker": None,
        "coverage": {"requested_obligations": [], "mpi_topology": None, "witnesses": {}},
        "mpi": False,
        "mpi_ranks": 1,
        "mpi_topology": None,
        "timeline": timeline,
        "amr_diagnostics": amr_diagnostics,
    }
    identity_inputs = {
        "schema_version": SCHEMA_VERSION,
        "configuration": configuration,
        "method": method,
        "execution": execution,
        "source_fingerprint": source_fingerprint,
        "artifact": artifact,
    }
    result_identity = _canonical_sha256(identity_inputs)
    payload["schema_version"] = np.asarray(SCHEMA_VERSION)
    payload["result_identity"] = np.asarray(result_identity)
    data_path = tmp_path / ("timeline_%dd_ts9_rid%s.npz" % (dimension, result_identity[:16]))
    np.savez_compressed(data_path, **payload)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "result_identity": result_identity,
        "result_identity_inputs": identity_inputs,
        "source_fingerprint": source_fingerprint,
        "data": data_path.name,
        "data_sha256": digest,
        **configuration,
        "time_snapshots": len(times),
        "metrics": {
            "method": method,
            "qualification": {
                "kind": "run_integrity",
                "run_integrity_passed": True,
            },
            "time_history": {
                "time": times.tolist(),
                "mass": [1.0] * len(times),
                "mass_relative_drift": [0.0] * len(times),
                "amplitude_rms": [0.07] * len(times),
                "exact_amplitude_rms": [0.071] * len(times),
                "phase_error_cycles": (0.01 * times).tolist(),
                "l1": (0.001 * times).tolist(),
                "l2": (0.002 * times).tolist(),
                "linf": (0.003 * times).tolist(),
            },
        },
        "provenance": {
            "repository_sha": "test-revision",
            "repository_dirty": False,
            "pops_version": "test-version",
            "date_utc": "2026-08-20T00:00:00+00:00",
            "execution": execution,
            "source": {
                "repository_sha": "test-revision",
                "repository_dirty": False,
                "tracked_diff_sha256": "0" * 64,
                "files": {
                    "benchmarks/verification/advection/sine_wave/generate_data.py": "b" * 64
                },
                "build_tree": {
                    "schema_version": "pops.sine-wave.build-source-tree.v1",
                    "roots": [
                        "CMakeLists.txt",
                        "cmake",
                        "include",
                        "src",
                        "python",
                        "pyproject.toml",
                        "schemas",
                        "scripts",
                    ],
                    "files": {"CMakeLists.txt": "c" * 64},
                },
                "fingerprint": source_fingerprint,
            },
            "artifact": artifact,
            "campaign": {
                "mpi_ranks": 1,
                "time_snapshots": len(times),
                "timeline_times": times.tolist(),
            },
        },
    }
    build_tree = metadata["provenance"]["source"]["build_tree"]
    build_tree["fingerprint"] = _canonical_sha256(build_tree)
    metadata_path = data_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return data_path, metadata_path


def test_sine_wave_campaign_has_separate_data_and_plot_authorities():
    generator = (CASE / "generate_data.py").read_text(encoding="utf-8")
    support = SUPPORT.read_text(encoding="utf-8")
    plotter = (CASE / "plot_results.py").read_text(encoding="utf-8")
    assert (CASE / "README.md").is_file()
    assert SUPPORT.is_file()
    assert (CASE / "results" / ".gitignore").is_file()
    assert (CASE / "figures" / ".gitignore").is_file()
    assert "matplotlib" not in generator
    assert "import pops" not in plotter
    assert "allow_pickle=False" in plotter
    assert "matplotlib" not in _import_roots(generator)
    assert "generate_data" not in _import_roots(support)
    assert "pops" not in _import_roots(plotter)
    assert "generate_data" not in _import_roots(plotter)


def test_generator_records_the_fixed_wave_and_public_pops_lifecycle():
    source = (CASE / "generate_data.py").read_text(encoding="utf-8")
    support = SUPPORT.read_text(encoding="utf-8")
    for token in (
        "WAVE_NUMBERS = (1, 2, 3)",
        "PROBE_TIME = 0.37",
        "BASE_CFL = 0.40",
        '"--cycles"',
        '"--time-snapshots"',
        "DEFAULT_TIME_SNAPSHOTS = 17",
        'SCHEMA_VERSION = "pops.sine-wave.v3"',
        "q_state = model.state(",
        '"U",',
        "pops.Model",
        "pops.Case",
        "pops.validate",
        "pops.resolve",
        "pops.compile",
        "pops.bind",
        "pops.run",
        'simulation.integral("tracer")',
        "SSPRK2(tracer_q, rate=rate)",
        "PrescribedWindow(",
        "Coarsen(~prescribed_patch)",
        '"constant_velocity_layout_periodicity"',
        '"--obligation"',
        "AMRRegrid.frozen()",
        "Coarsen(",
        "BergerRigoutsos(maximum_box_size=args.block_size)",
        "ExecutionContext.mpi_world",
        "RegularBlocks(max_cells=args.block_size)",
        '"mpi_ranks"',
    ):
        assert token in source
    for token in (
        '"schema_version"',
        '"data_sha256"',
        '"result_identity"',
        '"source_fingerprint"',
        '"cfl_formula": "base_cfl / dimension"',
        '"run_integrity_passed"',
        '"phase_error_cycles"',
        "state_global",
        "block_level_state_global",
        "patch_boxes",
        "local_boxes",
        '"has_coarse_fine_interface"',
        '"interface_mask"',
        "simulation.amr.explain_regrid()",
        '"topology_epoch"',
        "os.link(temporary_data_path, data_path)",
    ):
        assert token in support
    assert "_executor" not in source
    assert "AmrSystem" not in source
    assert "MAX_DT" not in source
    assert source.count("pops.run(") == 1
    for forbidden in (
        "hashlib",
        "importlib",
        "json",
        "platform",
        "subprocess",
        "PurePosixPath",
        "os.link",
    ):
        assert forbidden not in source
    generator_tree = ast.parse(source)
    function_names = {
        node.name for node in generator_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert function_names == {
        "_effective_cfl",
        "_parse_resolution",
        "_arguments",
        "main",
    }
    support_function_names = {
        node.name for node in ast.parse(support).body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "collect_snapshot",
        "compute_metrics",
        "coverage_witnesses",
        "execution_provenance",
        "native_receipt",
        "source_provenance",
        "publish_result",
    } <= support_function_names
    markers = [source.index("# %d." % section) for section in range(1, 10)]
    assert markers == sorted(markers)
    lifecycle_lines = {}
    for node in ast.walk(generator_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pops"
            and node.func.attr in {"validate", "resolve", "compile", "bind", "run"}
        ):
            lifecycle_lines[node.func.attr] = node.lineno
    lifecycle = ("validate", "resolve", "compile", "bind", "run")
    assert set(lifecycle_lines) == set(lifecycle)
    assert [lifecycle_lines[name] for name in lifecycle] == sorted(lifecycle_lines.values())
    calls = {
        node.func.id
        for node in ast.walk(generator_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "Tag" in calls
    assert "Coarsen" in calls


def test_prescribed_window_case_reaches_compile_without_running_numerics(monkeypatch):
    import pops

    class ResolvedCaseReachedCompile(Exception):
        pass

    def stop_before_compile(resolved):
        from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

        class NativeTaggingProbe:
            arguments = None

            def _set_bootstrap_tagging(self, *arguments):
                self.arguments = arguments

        assert resolved.resolved_hierarchy.plan.level_count == 2
        probe = NativeTaggingProbe()
        flow_bootstrap_tagging(
            probe,
            resolved.bootstrap_plan,
            {},
            clock_identity=resolved.time.clock.qualified_id,
            field_plans=resolved.field_plans,
        )
        assert probe.arguments is not None
        assert probe.arguments[0] == ["geometry", "geometry"]
        assert probe.arguments[8] == [
            [0.25, 0.18, 0.0],
            [0.25, 0.18, 0.0],
        ]
        raise ResolvedCaseReachedCompile

    scope = runpy.run_path(str(CASE / "generate_data.py"))
    monkeypatch.setattr(pops, "compile", stop_before_compile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--dimension",
            "1",
            "--resolution",
            "64",
            "--mode",
            "x",
            "--layout",
            "amr-frozen",
            "--subcycling",
            "synchronous",
            "--block-size",
            "16",
            "--cycles",
            "1",
            "--time-snapshots",
            "17",
        ],
    )

    with pytest.raises(ResolvedCaseReachedCompile):
        scope["main"]()


def test_effective_cfl_is_dimensionally_conservative():
    scope = runpy.run_path(str(CASE / "generate_data.py"))
    effective_cfl = scope["_effective_cfl"]

    assert effective_cfl(1) == pytest.approx(0.4)
    assert effective_cfl(2) == pytest.approx(0.2)
    assert effective_cfl(3) == pytest.approx(0.4 / 3.0)


def _native_provenance_fixture(tmp_path: Path) -> tuple[types.SimpleNamespace, Path]:
    """Make a byte-authenticated selected extension without loading a real shared object."""
    native_root = tmp_path / "site-packages" / "pops" / "_native"
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    extension = native_root / "dim2" / ("_pops" + suffix)
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"authenticated-native-extension")
    extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    abi_key = (
        "compiler=Apple LLVM test;std=202002L;headers="
        + "b" * 64
        + ";kokkos=1;stdlib=libc++_test;mpi=0;mpi_abi=off;dim=2"
    )
    row = {
        "dimension": 2,
        "path": "dim2/_pops" + suffix,
        "sha256": extension_sha256,
        "version": "test-version",
        "abi_key": abi_key,
        "build_fingerprint": "a" * 64,
        "has_mpi": False,
        "has_kokkos": True,
    }
    (native_root / "variants.json").write_text(
        json.dumps({"schema_version": 2, "variants": [row]}), encoding="utf-8"
    )
    module = types.SimpleNamespace(
        __file__=str(extension),
        __native_dimension__=2,
        __version__="test-version",
        abi_key=lambda: abi_key,
        __build_fingerprint__="a" * 64,
        __has_mpi__=False,
        __has_kokkos__=True,
    )
    return module, extension


def _source_provenance_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repository"
    (root / "CMakeLists.txt").parent.mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("# test CMake\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    for directory in ("cmake", "include", "src", "python", "schemas", "scripts"):
        source_directory = root / directory
        source_directory.mkdir()
        (source_directory / "source.txt").write_text("# test %s\n" % directory, encoding="utf-8")
    generator = (
        root / "benchmarks" / "verification" / "advection" / "sine_wave" / "generate_data.py"
    )
    generator.parent.mkdir(parents=True)
    generator.write_text("# test generator\n", encoding="utf-8")
    support = generator.with_name("_case_support.py")
    support.write_text("# test support\n", encoding="utf-8")
    helpers = root / "helpers"
    verification = helpers / "verification"
    verification.mkdir(parents=True)
    (helpers / "__init__.py").write_text("# helpers\n", encoding="utf-8")
    (verification / "__init__.py").write_text("# verification\n", encoding="utf-8")
    (verification / "sine_wave.py").write_text("# sine wave\n", encoding="utf-8")
    return root, generator, support


def test_source_provenance_v2_hashes_imported_initializers_and_is_deterministic(
    tmp_path, monkeypatch
):
    scope = _support_scope()
    source_provenance = scope["source_provenance"]
    monkeypatch.setitem(scope, "_git_value", lambda *_arguments: "test-revision")
    monkeypatch.setitem(scope, "_git_content_sha256", lambda *_arguments: "d" * 64)
    root, generator, support = _source_provenance_fixture(tmp_path)
    module, _ = _native_provenance_fixture(tmp_path)

    first = source_provenance(
        repository_root=root,
        generator_path=generator,
        support_path=support,
        source_schema_version="pops.sine-wave.source.v2",
        native_module=module,
    )
    second = source_provenance(
        repository_root=root,
        generator_path=generator,
        support_path=support,
        source_schema_version="pops.sine-wave.source.v2",
        native_module=module,
    )
    assert first == second
    assert first["schema_version"] == "pops.sine-wave.source.v2"
    assert set(first["files"]) == {
        "benchmarks/verification/advection/sine_wave/generate_data.py",
        "benchmarks/verification/advection/sine_wave/_case_support.py",
        "helpers/__init__.py",
        "helpers/verification/__init__.py",
        "helpers/verification/sine_wave.py",
    }

    (root / "helpers" / "__init__.py").write_text("# changed helpers\n", encoding="utf-8")
    changed = source_provenance(
        repository_root=root,
        generator_path=generator,
        support_path=support,
        source_schema_version="pops.sine-wave.source.v2",
        native_module=module,
    )
    assert changed["fingerprint"] != first["fingerprint"]

    # This file is deliberately absent from the synthetic Git fixture: it models
    # an untracked CMake source which nevertheless changes the native build.
    untracked_cmake = root / "cmake" / "PopsNativeBuildFingerprint.cmake"
    untracked_cmake.write_text("set(POPS_TEST_UNTRACKED on)\n", encoding="utf-8")
    changed_untracked_build = source_provenance(
        repository_root=root,
        generator_path=generator,
        support_path=support,
        source_schema_version="pops.sine-wave.source.v2",
        native_module=module,
    )
    assert (
        changed_untracked_build["build_tree"]["fingerprint"]
        != changed["build_tree"]["fingerprint"]
    )


def test_native_provenance_receipt_is_canonical_and_has_no_absolute_path(tmp_path):
    native_receipt = _support_scope()["native_receipt"]
    module, _ = _native_provenance_fixture(tmp_path)

    receipt = native_receipt(module)

    assert receipt["path"].startswith("dim2/_pops")
    assert receipt["dimension"] == 2
    assert receipt["has_mpi"] is False
    assert receipt["has_kokkos"] is True
    assert str(tmp_path) not in json.dumps(receipt, sort_keys=True)


def test_native_provenance_refuses_extension_bytes_that_differ_from_manifest(tmp_path):
    native_receipt = _support_scope()["native_receipt"]
    module, extension = _native_provenance_fixture(tmp_path)
    extension.write_bytes(b"tampered-native-extension")

    with pytest.raises(RuntimeError, match="bytes differ"):
        native_receipt(module)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    (
        ("__native_dimension__", 3, "dimension differs"),
        ("__version__", "tampered-version", "version differs"),
        ("__has_mpi__", True, "backend facts differ"),
    ),
)
def test_native_provenance_refuses_manifest_row_and_module_fact_mismatches(
    tmp_path, attribute, value, message
):
    native_receipt = _support_scope()["native_receipt"]
    module, _ = _native_provenance_fixture(tmp_path)
    setattr(module, attribute, value)

    with pytest.raises(RuntimeError, match=message):
        native_receipt(module)


def test_uniform_layout_accepts_an_explicit_regular_block_size(monkeypatch):
    arguments = runpy.run_path(str(CASE / "generate_data.py"))["_arguments"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--dimension",
            "2",
            "--layout",
            "uniform",
            "--block-size",
            "8",
        ],
    )

    args = arguments()

    assert args.layout == "uniform"
    assert args.block_size == 8


def test_versioned_matrix_fail_closes_coverage_and_never_executes_by_default(capsys):
    scope = runpy.run_path(str(MATRIX_DRIVER))
    matrix = scope["_read_matrix"](MATRIX)
    cases = scope["_validate_matrix"](matrix)

    assert scope["REPOSITORY_ROOT"] == REPOSITORY_ROOT
    assert scope["BUILD_SCRIPT"] == REPOSITORY_ROOT / "scripts" / "build_python.sh"
    assert scope["BUILD_SCRIPT"].is_file()
    assert scope["_matrix_output_root"](
        Path("benchmarks/results"), "matrix-v1"
    ) == REPOSITORY_ROOT / "benchmarks/results/matrix-v1"
    assert {case["dimension"] for case in cases} == {1, 2, 3}
    assert {case["layout"] for case in cases} == {"uniform", "amr-frozen", "amr-mobile"}
    assert {case["mpi_ranks"] for case in cases if case["mpi"]} == {1, 2, 4, 8}
    assert set(matrix["coverage_obligations"]) == scope["KNOWN_OBLIGATIONS"]
    for obligation, declaration in matrix["coverage_obligations"].items():
        witnessed = {case["id"] for case in cases if obligation in case["obligations"]}
        assert witnessed == set(declaration["case_ids"])
    repeated = next(case for case in cases if case["id"] == "d2-cf-subcycled")
    assert repeated["layout"] == "amr-frozen"
    assert repeated["cycles"] == 3 and repeated["time_snapshots"] == 49
    assert "repeated_patch_crossing" in repeated["obligations"]
    mobile = next(case for case in cases if case["id"] == "d2-mobile-subcycled")
    assert mobile["cycles"] == 1
    assert mobile["obligations"] == ["prescribed_mobile_regrid"]

    command = scope["_command"](
        next(case for case in cases if case["id"] == "d3-edge"), Path("out")
    )
    assert "--mode" in command and command[command.index("--mode") + 1] == "xy"
    assert "--obligation" in command
    assert "pops.run(" not in MATRIX_DRIVER.read_text(encoding="utf-8")
    assert "subprocess.run(" in MATRIX_DRIVER.read_text(encoding="utf-8")
    assert scope["_resolved_resolution"](
        next(case for case in cases if case["id"] == "d1-face")
    ) == [64]
    assert scope["_resolved_resolution"](
        next(case for case in cases if case["id"] == "d2-face")
    ) == [64, 64]
    assert scope["_resolved_resolution"](
        next(case for case in cases if case["id"] == "d3-face")
    ) == [32, 32, 32]
    assert capsys.readouterr().out == ""


def test_individual_convergence_witness_is_explicitly_deferred():
    witness = _support_scope()["coverage_witnesses"]
    values = witness(
        dimension=1,
        velocity=(1.0,),
        layout="uniform",
        resolution=(32,),
        mode="x",
        cycles=1,
        obligations=("second_order_convergence",),
        timeline_times=(0.0, 0.5, 1.0),
        timeline_snapshots=(
            {
                "local_boxes": (((0,), (16,)), ((16,), (31,))),
                "regrid_count": 0,
                "topology_epoch": 0,
            },
            {
                "local_boxes": (((0,), (16,)), ((16,), (31,))),
                "regrid_count": 0,
                "topology_epoch": 0,
            },
            {
                "local_boxes": (((0,), (16,)), ((16,), (31,))),
                "regrid_count": 0,
                "topology_epoch": 0,
            },
        ),
        metrics={"qualification": {"coarse_fine_interface_seen": False}},
        witness_reference_point=(0.137,),
        patch_velocity=(0.5,),
        wave_numbers=(1,),
        final_time=1.0,
        refinement_ratio=2,
    )
    driver = runpy.run_path(str(MATRIX_DRIVER))

    assert values["second_order_convergence"] == driver["DEFERRED_CONVERGENCE_WITNESS"]
    assert values["second_order_convergence"]["observed"] is False


def test_pair_verification_rejects_forged_or_non_deferred_convergence_witnesses():
    driver = runpy.run_path(str(MATRIX_DRIVER))
    expected = driver["DEFERRED_CONVERGENCE_WITNESS"]
    observed = json.loads(json.dumps(expected))
    observed["observed"] = True
    non_deferred = json.loads(json.dumps(expected))
    non_deferred.pop("deferred")

    assert driver["_is_deferred_convergence_witness"](expected) is True
    assert driver["_is_deferred_convergence_witness"](observed) is False
    assert driver["_is_deferred_convergence_witness"](non_deferred) is False


def test_matrix_convergence_receipt_requires_each_case_to_remain_deferred():
    driver = runpy.run_path(str(MATRIX_DRIVER))
    declaration = {
        "case_ids": ["n32", "n64", "n128"],
        "reported_norms": ["l1", "l2", "linf"],
        "qualified_norm": "l1",
        "minimum_order": 1.75,
    }
    metadata = {}
    for identifier, resolution in zip(declaration["case_ids"], (32, 64, 128), strict=True):
        error = resolution**-2
        metadata[identifier] = {
            "source_fingerprint": "f" * 64,
            "resolution": [resolution],
            "coverage": {
                "requested_obligations": ["second_order_convergence"],
                "witnesses": {
                    "second_order_convergence": driver["DEFERRED_CONVERGENCE_WITNESS"]
                },
            },
            "provenance": {"execution": {"runtime": {"kokkos_backend": "OpenMP"}}},
            "metrics": {
                "method": {"time": "SSPRK2", "reconstruction": "MUSCL(VanLeer)"},
                "errors": {"l1": error, "l2": error, "linf": error},
            },
        }
    matrix = {"convergence_series": {"dim1": declaration}}

    receipt = driver["_convergence_receipt"](matrix, metadata)
    assert receipt["dim1"]["orders"]["l1"] == pytest.approx([2.0, 2.0])

    forged = json.loads(json.dumps(metadata))
    forged["n32"]["coverage"]["witnesses"]["second_order_convergence"]["observed"] = True
    with pytest.raises(RuntimeError, match="lacks the explicit deferred matrix receipt"):
        driver["_convergence_receipt"](matrix, forged)

    malformed = json.loads(json.dumps(metadata))
    malformed["n32"]["coverage"]["witnesses"]["second_order_convergence"].pop("deferred")
    with pytest.raises(RuntimeError, match="lacks the explicit deferred matrix receipt"):
        driver["_convergence_receipt"](matrix, malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("repository_sha", "other-revision"),
        ("tracked_diff_sha256", "1" * 64),
    ),
)
def test_complete_matrix_refuses_mixed_non_native_source_authority(tmp_path, field, value):
    """Native receipts may vary by build phase; repository source authority may not."""
    scope = runpy.run_path(str(MATRIX_DRIVER))
    output = tmp_path / "matrix"
    first_dir = output / "first"
    second_dir = output / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    first_data, first_metadata_path = _write_timeline_case(first_dir, 1)
    second_data, second_metadata_path = _write_timeline_case(second_dir, 2)
    first_metadata = json.loads(first_metadata_path.read_text(encoding="utf-8"))
    second_metadata = json.loads(second_metadata_path.read_text(encoding="utf-8"))
    second_metadata["provenance"]["source"][field] = value
    second_metadata_path.write_text(json.dumps(second_metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="repository revision, tracked diff, source files"):
        scope["_write_complete_manifest"](
            matrix_path=MATRIX,
            output_root=output,
            cases=(
                {"id": "first", "dimension": 2, "mpi": False},
                {"id": "second", "dimension": 2, "mpi": False},
            ),
            results={
                "first": (first_data, first_metadata_path, first_metadata),
                "second": (second_data, second_metadata_path, second_metadata),
            },
            convergence={},
        )


def test_complete_matrix_refuses_a_relevant_untracked_build_source_mutation(tmp_path, monkeypatch):
    """An untracked CMake leaf is source authority, not ignorable build debris."""
    matrix_scope = runpy.run_path(str(MATRIX_DRIVER))
    support_scope = _support_scope()
    monkeypatch.setitem(support_scope, "_git_value", lambda *_arguments: "test-revision")
    monkeypatch.setitem(support_scope, "_git_content_sha256", lambda *_arguments: "0" * 64)
    root, generator, support_path = _source_provenance_fixture(tmp_path)
    module, _ = _native_provenance_fixture(tmp_path)
    first_source = support_scope["source_provenance"](
        repository_root=root,
        generator_path=generator,
        support_path=support_path,
        source_schema_version="pops.sine-wave.source.v2",
        native_module=module,
    )
    (root / "cmake" / "PopsNativeBuildFingerprint.cmake").write_text(
        "set(POPS_TEST_UNTRACKED on)\n", encoding="utf-8"
    )
    second_source = support_scope["source_provenance"](
        repository_root=root,
        generator_path=generator,
        support_path=support_path,
        source_schema_version="pops.sine-wave.source.v2",
        native_module=module,
    )
    assert first_source["build_tree"]["fingerprint"] != second_source["build_tree"]["fingerprint"]
    # The fixture is intentionally not a Git repository; supply the stable Git
    # receipt which a real campaign records, keeping the tree mutation as the
    # sole source-authority difference under test.
    for source in (first_source, second_source):
        source["repository_sha"] = "test-revision"
        source["tracked_diff_sha256"] = "0" * 64

    output = tmp_path / "matrix"
    first_dir = output / "first"
    second_dir = output / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir()
    first_data, first_metadata_path = _write_timeline_case(first_dir, 1)
    second_data, second_metadata_path = _write_timeline_case(second_dir, 2)
    first_metadata = json.loads(first_metadata_path.read_text(encoding="utf-8"))
    second_metadata = json.loads(second_metadata_path.read_text(encoding="utf-8"))
    first_metadata["provenance"]["source"] = first_source
    second_metadata["provenance"]["source"] = second_source

    with pytest.raises(RuntimeError, match="repository revision, tracked diff, source files"):
        matrix_scope["_write_complete_manifest"](
            matrix_path=MATRIX,
            output_root=output,
            cases=(
                {"id": "first", "dimension": 2, "mpi": False},
                {"id": "second", "dimension": 2, "mpi": False},
            ),
            results={
                "first": (first_data, first_metadata_path, first_metadata),
                "second": (second_data, second_metadata_path, second_metadata),
            },
            convergence={},
        )


def test_matrix_locks_the_exact_inventory_and_controlled_comparisons():
    scope = runpy.run_path(str(MATRIX_DRIVER))
    matrix = scope["_read_matrix"](MATRIX)
    cases = scope["_validate_matrix"](matrix)
    by_id = {case["id"]: case for case in cases}

    assert tuple(by_id) == scope["EXPECTED_CASE_IDS"]
    assert len(cases) == 37
    for synchronous, subcycled in scope["SUBCYCLING_COMPARISON_PAIRS"]:
        assert scope["_same_configuration"](
            by_id[synchronous], by_id[subcycled], ignored={"subcycling"}
        )
    mpi = [by_id[identifier] for identifier in scope["MPI_INVARIANCE_CASE_IDS"]]
    assert [row["mpi_ranks"] for row in mpi] == [1, 2, 4]
    assert all(
        scope["_same_configuration"](mpi[0], row, ignored={"mpi_ranks"}) for row in mpi[1:]
    )
    corner_mpi = by_id["d3-mpi-np8-corner"]
    assert corner_mpi["mpi_ranks"] == 8
    assert corner_mpi["mpi_topology"] == [2, 2, 2]
    assert "block_corner_3d" in corner_mpi["obligations"]
    blocks = [by_id[identifier] for identifier in scope["BLOCK_SIZE_COMPARISON_CASE_IDS"]]
    assert [row["block_size"] for row in blocks] == [8, 16, 32]
    assert all(
        scope["_same_configuration"](blocks[0], row, ignored={"block_size"})
        for row in blocks[1:]
    )

    malformed = json.loads(json.dumps(matrix))
    next(row for row in malformed["cases"] if row["id"] == "d3-mobile-subcycled")[
        "block_size"
    ] = 16
    with pytest.raises(ValueError, match="d3-mobile-sync/d3-mobile-subcycled"):
        scope["_validate_matrix"](malformed)


def test_matrix_source_is_hash_sealed_before_validation(tmp_path):
    scope = runpy.run_path(str(MATRIX_DRIVER), run_name="sine_wave_matrix_hash_contract")
    canonical = MATRIX.read_bytes()
    assert hashlib.sha256(canonical).hexdigest() == scope["CANONICAL_MATRIX_SHA256"]

    changed = tmp_path / MATRIX.name
    changed.write_bytes(canonical + b"\n")
    with pytest.raises(ValueError, match="exact canonical 37-case scientific inventory"):
        scope["_read_matrix"](changed)


def test_crossing_witness_requires_native_boxes_and_a_timeline_interval():
    scope = runpy.run_path(str(CASE / "generate_data.py"))
    arguments = scope["_arguments"]
    witness = _support_scope()["coverage_witnesses"]
    original = sys.argv
    try:
        sys.argv = [
            "generate_data.py",
            "--dimension",
            "3",
            "--resolution",
            "32",
            "--mode",
            "diagonal",
            "--layout",
            "uniform",
            "--block-size",
            "8",
            "--obligation",
            "block_corner_3d",
        ]
        args = arguments()
    finally:
        sys.argv = original
    boxes = tuple(
        ((x, y, z), (x + 8, y + 8, z + 8))
        for x in range(0, 32, 8)
        for y in range(0, 32, 8)
        for z in range(0, 32, 8)
    )
    snapshots = tuple(
        {"local_boxes": boxes, "regrid_count": 0, "topology_epoch": 0} for _ in range(17)
    )
    values = witness(
        dimension=args.dimension,
        velocity=args.velocity,
        layout=args.layout,
        resolution=args.resolution,
        mode=args.mode,
        cycles=args.cycles,
        obligations=tuple(args.obligation),
        timeline_times=tuple(np.linspace(0.0, 1.0, 17)),
        timeline_snapshots=snapshots,
        metrics={"qualification": {"coarse_fine_interface_seen": False}},
        witness_reference_point=scope["WITNESS_REFERENCE_POINT"],
        patch_velocity=scope["PATCH_VELOCITY"],
        wave_numbers=scope["WAVE_NUMBERS"][: args.dimension],
        final_time=scope["FINAL_TIME"] * args.cycles,
        refinement_ratio=2,
    )

    assert values["block_corner_3d"]["observed"] is True
    assert len(values["block_corner_3d"]["planes"]) == 3


def test_static_amr_witness_requires_repeated_native_fine_box_crossings():
    witness = _support_scope()["coverage_witnesses"]
    times = tuple(np.linspace(0.0, 3.0, 49))
    static_boxes = ((1, (14, 14), (28, 32)),)
    snapshots = tuple(
        {
            "patch_rows": static_boxes,
            "regrid_count": 0,
            "topology_epoch": 0,
        }
        for _ in times
    )
    values = witness(
        dimension=2,
        velocity=(1.0, 1.0),
        layout="amr-frozen",
        resolution=(64, 64),
        mode="diagonal",
        cycles=3,
        obligations=("repeated_patch_crossing",),
        timeline_times=times,
        timeline_snapshots=snapshots,
        metrics={"qualification": {"coarse_fine_interface_seen": True}},
        witness_reference_point=(0.137, 0.137),
        patch_velocity=(0.5, 0.25),
        wave_numbers=(1, 2),
        final_time=3.0,
        refinement_ratio=2,
    )

    repeated = values["repeated_patch_crossing"]
    assert repeated["observed"] is True
    assert repeated["entries"] == repeated["exits"] == 3
    assert len(repeated["transition_brackets"]) == 6
    assert repeated["native_fine_boxes"] == [{"level": 1, "lower": [14, 14], "upper": [28, 32]}]

    changed = list(snapshots)
    changed[20] = {**changed[20], "patch_rows": ((1, (15, 14), (28, 32)),)}
    rejected = witness(
        dimension=2,
        velocity=(1.0, 1.0),
        layout="amr-frozen",
        resolution=(64, 64),
        mode="diagonal",
        cycles=3,
        obligations=(),
        timeline_times=times,
        timeline_snapshots=tuple(changed),
        metrics={"qualification": {"coarse_fine_interface_seen": True}},
        witness_reference_point=(0.137, 0.137),
        patch_velocity=(0.5, 0.25),
        wave_numbers=(1, 2),
        final_time=3.0,
        refinement_ratio=2,
    )
    assert rejected["repeated_patch_crossing"]["observed"] is False


def test_prescribed_window_rejects_a_same_rank_but_different_layout_frame():
    from pops.amr import PrescribedWindow
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian
    from pops.time import Clock

    frame = CartesianDomain("window_a", (0.0,), (1.0,)).frame(Cartesian(1))
    different_frame = CartesianDomain("window_b", (0.0,), (1.0,)).frame(Cartesian(1))
    clock = Clock("window")
    window = PrescribedWindow(
        frame=frame, clock=clock, center=(0.25,), half_width=(0.1,), velocity=(1.0,)
    )

    with pytest.raises(ValueError, match="frame differs"):
        window.resolve_for_amr_predicate(
            types.SimpleNamespace(dimension=1, frame_id=different_frame.canonical_id, clock=clock),
            action="refine",
        )
    assert (
        window.resolve_for_amr_predicate(
            types.SimpleNamespace(dimension=1, frame_id=frame.canonical_id, clock=clock),
            action="coarsen",
        )
        is window
    )
    assert window.canonical_identity()["trajectory"] == "constant_velocity_layout_periodicity"


def test_plotter_composes_coarse_and_fine_leaf_cells_over_the_whole_domain():
    field_pair = runpy.run_path(str(CASE / "plot_results.py"))["_field_pair"]
    coarse = np.array([[1.0, 2.0], [3.0, 4.0]])
    fine = np.arange(16, dtype=float).reshape(4, 4) + 10.0
    fine_mask = np.zeros((4, 4), dtype=bool)
    fine_mask[:2, 2:] = True
    data = {
        "numeric_level_0": coarse,
        "exact_level_0": coarse + 100.0,
        "mask_level_0": np.array([[True, False], [True, True]]),
        "numeric_level_1": fine,
        "exact_level_1": fine + 100.0,
        "mask_level_1": fine_mask,
    }

    numerical, exact, mask, level = field_pair(data)

    assert level == 1
    assert mask.all()
    assert np.array_equal(numerical[:2, :2], np.ones((2, 2)))
    assert np.array_equal(numerical[:2, 2:], fine[:2, 2:])
    assert np.array_equal(exact, numerical + 100.0)


def test_plotter_rejects_a_json_from_another_npz(tmp_path):
    load = runpy.run_path(str(CASE / "plot_results.py"))["_load"]
    data_path = tmp_path / "run.npz"
    np.savez_compressed(
        data_path,
        numeric=np.ones((4, 4)),
        exact=np.ones((4, 4)),
        mask=np.ones((4, 4), dtype=bool),
    )
    metadata_path = tmp_path / "run.json"
    metadata_path.write_text(
        json.dumps({"data": "other.npz", "dimension": 2, "resolution": [4, 4]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authenticate"):
        load(data_path, metadata_path)


def test_plotter_rejects_stale_schema_and_content_digest(tmp_path):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    load = scope["_load"]
    data_path, metadata_path = _write_timeline_case(tmp_path, 1)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata["schema_version"] = "pops.sine-wave.v2"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load(data_path, metadata_path)

    metadata["schema_version"] = SCHEMA_VERSION
    metadata["data_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="content digest"):
        load(data_path, metadata_path)


@pytest.mark.parametrize(
    "mutation",
    ("backend", "source", "snapshots"),
)
def test_plotter_rejects_tampered_identity_provenance(tmp_path, mutation):
    load = runpy.run_path(str(CASE / "plot_results.py"))["_load"]
    data_path, metadata_path = _write_timeline_case(tmp_path, 2)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "backend":
        metadata["result_identity_inputs"]["execution"]["runtime"]["kokkos_backend"] = "Cuda"
    elif mutation == "source":
        metadata["source_fingerprint"] = "b" * 64
    else:
        metadata["provenance"]["campaign"]["time_snapshots"] = 10
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="identity|provenance"):
        load(data_path, metadata_path)


def test_timeline_validator_requires_nine_consistent_exact_times(tmp_path):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    data_path, metadata_path = _write_timeline_case(tmp_path, 2)
    data, metadata = scope["_load"](data_path, metadata_path)
    data["timeline_time"] = data["timeline_time"].copy()
    data["timeline_time"][4] += 1.0e-5

    with pytest.raises(ValueError, match="timeline times"):
        scope["_timeline_frames"](data, metadata)


def test_timeline_composes_changing_nested_visualization_shapes(tmp_path):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    data_path, metadata_path = _write_timeline_case(tmp_path, 2)
    data, metadata = scope["_load"](data_path, metadata_path)
    prefix = "timeline_0000_"
    coarse_numeric = data.pop(prefix + "numeric")[::2, ::2]
    coarse_exact = data.pop(prefix + "exact")[::2, ::2]
    data.pop(prefix + "mask")
    data.pop(prefix + "interface_mask")
    data.pop(prefix + "x")
    data.pop(prefix + "y")
    data[prefix + "numeric_level_0"] = coarse_numeric
    data[prefix + "exact_level_0"] = coarse_exact
    data[prefix + "mask_level_0"] = np.ones_like(coarse_exact, dtype=bool)
    data[prefix + "interface_mask_level_0"] = np.zeros_like(coarse_exact, dtype=bool)
    data[prefix + "x_level_0"] = (np.arange(3, dtype=float) + 0.5) / 3.0
    data[prefix + "y_level_0"] = (np.arange(3, dtype=float) + 0.5) / 3.0

    _, frames = scope["_timeline_frames"](data, metadata)

    assert all(frame["numeric"].shape == (6, 6) for frame in frames)
    assert frames[0]["native_shape"] == (3, 3)
    assert np.array_equal(frames[0]["numeric"][:2, :2], np.full((2, 2), coarse_numeric[0, 0]))


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (1, {"profile_1d.png", "evolution_1d.gif"}),
        (2, {"fields_2d.png", "evolution_2d.gif"}),
        (
            3,
            {
                "cuts_3d.png",
                "storyboard_3d_cuts.png",
                "isosurface_3d.png",
                "oblique_cut_3d.png",
                "evolution_3d.gif",
            },
        ),
    ),
)
def test_complete_renderer_primitives_cover_1d_2d_and_3d(tmp_path, dimension, expected):
    from PIL import Image
    import matplotlib

    scope = runpy.run_path(str(CASE / "plot_results.py"))
    data_path, metadata_path = _write_timeline_case(tmp_path, dimension)
    figures = tmp_path / ("figures_%dd" % dimension)
    figures.mkdir()
    data, metadata = scope["_load"](data_path, metadata_path)
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    generated = scope["_render_run"](plt, data, metadata, figures=figures, fps=20)

    names = {path.name for path in generated}
    assert expected <= names
    animation_path = next(path for path in generated if path.suffix == ".gif")
    with Image.open(animation_path) as animation:
        assert animation.n_frames == 9


def test_3d_storyboard_and_slice_fallback_use_cell_centers():
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    assert np.array_equal(
        scope["_cell_centers"](4),
        np.asarray((0.125, 0.375, 0.625, 0.875)),
    )

    class Axis:
        def __init__(self):
            self.calls = []

        def contour(self, x, y, plane, **kwargs):
            self.calls.append((np.asarray(x), np.asarray(y), np.asarray(plane), kwargs))

    axis = Axis()
    field = np.tile(np.asarray(((-1.0, 1.0), (-1.0, 1.0))), (4, 1, 1))
    scope["_slice_contour_surface"](axis, field, 0.0, color="black")

    assert [call[3]["offset"] for call in axis.calls] == pytest.approx((0.125, 0.375, 0.625, 0.875))
    for x, y, _, _ in axis.calls:
        assert np.array_equal(x, np.asarray((0.25, 0.75)))
        assert np.array_equal(y, np.asarray((0.25, 0.75)))


def test_amr_interface_and_regrid_diagnostics_are_data_only_and_observation_labeled(
    tmp_path,
):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = np.linspace(0.0, 1.0, 9)
    interface = np.zeros((4, 4), dtype=bool)
    interface[:, 1] = True
    frames = []
    for index, time in enumerate(times):
        exact = np.ones((4, 4))
        numeric = exact + (0.001 + time * 0.01) * np.arange(16).reshape(4, 4)
        frames.append(
            {
                "time": float(time),
                "numeric": numeric,
                "exact": exact,
                "interface": interface,
                "patches": np.asarray(((0, 0, 1, 3, 1),), dtype=np.int64),
                "regrid_count": 0 if index < 3 else (1 if index < 7 else 3),
                "topology_epoch": 0 if index < 3 else (1 if index < 7 else 2),
            }
        )
    metadata = {
        "layout": "amr-mobile",
        "amr_diagnostics": _test_amr_diagnostics(),
    }

    interface_target = tmp_path / "interface_vs_bulk_error.png"
    regrid_target = tmp_path / "regrid_events.png"
    assert (
        scope["_plot_interface_vs_bulk_error"](plt, frames, metadata, interface_target)
        == interface_target
    )
    assert scope["_plot_regrid_events"](plt, frames, metadata, regrid_target) == regrid_target
    assert interface_target.is_file()
    assert regrid_target.is_file()
    events = scope["_observed_regrid_events"](frames)
    assert [event[0] for event in events] == pytest.approx((times[3], times[7]))
    assert [event[1] for event in events] == [1, 2]


def test_campaign_emits_only_uniquely_compatible_amr_comparisons(tmp_path):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data_path, metadata_path = _write_timeline_case(tmp_path, 2)
    data, uniform = scope["_load"](data_path, metadata_path)
    uniform["block_size"] = 16
    synchronous = json.loads(json.dumps(uniform))
    synchronous["layout"] = "amr-mobile"
    synchronous["block_size"] = 8
    synchronous["patch_marker"] = {"velocity": [0.5, 0.25]}
    subcycled = json.loads(json.dumps(synchronous))
    subcycled["subcycling"] = "subcycled"
    output = tmp_path / "campaign"
    output.mkdir()

    generated = scope["_plot_amr_comparisons"](
        plt,
        [(data, uniform), (data, synchronous), (data, subcycled)],
        output,
    )

    assert {path.name for path in generated} == {
        "uniform_vs_amr.png",
        "subcycling_vs_nosubcycling.png",
    }
    assert all(path.is_file() for path in generated)
    stale = output / "uniform_vs_amr.png"
    assert scope["_plot_amr_comparisons"](plt, [(data, uniform)], output) == []
    # Comparison helpers now render only into a fresh publication staging directory.
    # They must never delete a media artifact that was already published.
    assert stale.is_file()


def test_timeline_fails_closed_without_interface_or_regrid_labels(tmp_path):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    data_path, metadata_path = _write_timeline_case(tmp_path, 2)
    data, metadata = scope["_load"](data_path, metadata_path)
    data.pop("timeline_0004_interface_mask")
    with pytest.raises(ValueError, match="interface mask"):
        scope["_timeline_frames"](data, metadata)

    data, metadata = scope["_load"](data_path, metadata_path)
    data.pop("timeline_0004_regrid_count")
    with pytest.raises(ValueError, match="regrid_count"):
        scope["_timeline_frames"](data, metadata)


def test_unqualified_or_incompatible_series_preserves_an_existing_convergence(tmp_path):
    plot_convergence = runpy.run_path(str(CASE / "plot_results.py"))["_plot_convergence"]
    stale = tmp_path / "convergence.png"
    stale.write_bytes(b"stale")
    execution = _test_execution_provenance()
    source_fingerprint = "a" * 64
    base = {
        "schema_version": SCHEMA_VERSION,
        "case": "periodic_sine_wave_advection",
        "dimension": 2,
        "mode": "diagonal",
        "wave_numbers": [1, 2],
        "velocity": [1.0, 1.0],
        "epsilon": 0.1,
        "probe_time": 0.37,
        "final_time": 1.0,
        "cycles": 1,
        "layout": "amr-mobile",
        "subcycling": "subcycled",
        "block_size": 8,
        "patch_marker": {"velocity": [0.5, 0.25]},
        "mpi": False,
        "mpi_ranks": 1,
        "timeline": {"frames": 9, "times": np.linspace(0.0, 1.0, 9).tolist()},
        "amr_diagnostics": _test_amr_diagnostics(),
        "provenance": {
            "pops_version": "test-version",
            "execution": execution,
            "source": {
                "repository_sha": "test-revision",
                "fingerprint": source_fingerprint,
            },
        },
        "result_identity_inputs": {
            "execution": execution,
            "source_fingerprint": source_fingerprint,
        },
        "resolution": [16, 16],
        "metrics": {
            "method": {"time": "SSPRK2", "cfl": 0.4},
            "qualification": {"passed": False},
            "probe_errors": {"l1": 0.1, "l2": 0.1, "linf": 0.1},
        },
    }
    other = json.loads(json.dumps(base))
    other["resolution"] = [32, 32]

    assert plot_convergence(None, [base, other], tmp_path) is None
    assert stale.read_bytes() == b"stale"


def test_composite_metrics_reject_a_leaf_mask_that_does_not_cover_unit_volume():
    metrics = _compute_metrics()
    shape = (4, 4)
    mask = np.ones(shape, dtype=bool)
    mask[0, 0] = False
    snapshot = {
        "numeric": np.ones(shape),
        "exact": np.ones(shape),
        "initial": np.ones(shape),
        "mask": mask,
        "patch_rows": (),
    }

    with pytest.raises(RuntimeError, match="represents volume"):
        metrics(
            snapshot,
            snapshot,
            snapshot,
            cells=(4, 4),
            layout="uniform",
            initial_mass=15.0 / 16.0,
            probe_mass=15.0 / 16.0,
            final_mass=15.0 / 16.0,
            final_time=1.0,
            probe_phase_cycles=0.37,
        )


def test_composite_metrics_reject_mass_disagreement_after_volume_passes():
    metrics = _compute_metrics()
    snapshot = {
        "numeric": np.ones(4),
        "exact": np.ones(4),
        "initial": np.ones(4),
        "mask": np.ones(4, dtype=bool),
        "patch_rows": (),
    }

    with pytest.raises(RuntimeError, match="disagrees with native integral"):
        metrics(
            snapshot,
            snapshot,
            snapshot,
            cells=(4,),
            layout="uniform",
            initial_mass=0.5,
            probe_mass=1.0,
            final_mass=1.0,
            final_time=1.0,
            probe_phase_cycles=0.37,
        )


def test_initial_diagnostics_use_the_native_initial_snapshot():
    metrics = _compute_metrics()
    mask = np.ones(4, dtype=bool)
    initial_snapshot = {
        "numeric": np.array([0.0, 0.0, 2.0, 2.0]),
        "exact": np.ones(4),
        "initial": np.ones(4),
        "mask": mask,
        "patch_rows": (),
    }
    later_snapshot = {
        "numeric": np.ones(4),
        "exact": np.ones(4),
        "initial": np.ones(4),
        "mask": mask,
        "patch_rows": (),
    }

    result = metrics(
        initial_snapshot,
        later_snapshot,
        later_snapshot,
        cells=(4,),
        layout="uniform",
        initial_mass=1.0,
        probe_mass=1.0,
        final_mass=1.0,
        final_time=1.0,
        probe_phase_cycles=0.37,
    )

    assert result["initial_diagnostics"]["amplitude_rms"] == pytest.approx(1.0)
    assert result["diagnostics"]["amplitude_rms"] == pytest.approx(0.0)


def test_time_history_uses_same_frame_exact_amplitude_and_all_mass_samples():
    metrics = _compute_metrics()
    x = (np.arange(32, dtype=float) + 0.5) / 32.0
    wave = np.sin(2.0 * np.pi * x)
    initial_reference = 1.0 + 0.1 * wave

    def snapshot(mean: float, numeric_amplitude: float, exact_amplitude: float):
        return {
            "numeric": mean + numeric_amplitude * wave,
            "exact": 1.0 + exact_amplitude * wave,
            "initial": initial_reference,
            "mask": np.ones(x.shape, dtype=bool),
            "patch_rows": (),
        }

    initial = snapshot(1.0, 0.1, 0.1)
    quarter = snapshot(1.0, 0.1, 0.2)
    hidden_mass_excursion = snapshot(1.2, 0.1, 0.2)
    final = snapshot(1.0, 0.15, 0.3)
    result = metrics(
        initial,
        final,
        quarter,
        cells=(32,),
        layout="uniform",
        initial_mass=1.0,
        probe_mass=1.0,
        final_mass=1.0,
        final_time=1.0,
        probe_phase_cycles=0.37,
        history_times=(0.0, 0.25, 0.75, 1.0),
        history_snapshots=(initial, quarter, hidden_mass_excursion, final),
        history_masses=(1.0, 1.0, 1.2, 1.0),
    )

    assert result["time_history"]["amplitude_retention"] == pytest.approx((1.0, 0.5, 0.5, 0.5))
    assert result["accuracy"]["final_amplitude_retention"] == pytest.approx(0.5)
    assert result["conservation"]["max_relative_drift"] == pytest.approx(0.2)
    assert result["conservation"]["timeline_samples"] == 4


def test_signed_phase_projection_distinguishes_a_lag_from_a_lead():
    phase_error = _support_scope()["_signed_phase_error_cycles"]
    x = (np.arange(128, dtype=float) + 0.5) / 128.0
    exact = 1.0 + 0.1 * np.sin(2.0 * np.pi * x)
    quadrature = 1.0 + 0.1 * np.cos(2.0 * np.pi * x)
    weights = np.full(x.shape, 1.0 / x.size)
    snapshot = {
        "quadrature": quadrature,
        "mask": np.ones(x.shape, dtype=bool),
    }

    lag = 1.0 + 0.1 * np.sin(2.0 * np.pi * (x - 0.075))
    lead = 1.0 + 0.1 * np.sin(2.0 * np.pi * (x + 0.075))

    assert phase_error(snapshot, lag, exact, weights, layout="uniform") == pytest.approx(0.075)
    assert phase_error(snapshot, lead, exact, weights, layout="uniform") == pytest.approx(-0.075)


def test_transport_probe_rejects_the_reverse_direction_and_accepts_the_oracle():
    metrics = _compute_metrics()
    cell_averages = _support_scope()["sine_wave_cell_averages"]
    cells = (64,)
    waves = (1,)
    initial, _ = cell_averages(cells, waves, epsilon=0.1)
    exact, _ = cell_averages(cells, waves, epsilon=0.1, displacement=(0.37,))
    reverse, _ = cell_averages(cells, waves, epsilon=0.1, displacement=(-0.37,))
    mask = np.ones(initial.shape, dtype=bool)

    def snapshot(numeric, reference):
        return {
            "numeric": numeric,
            "exact": reference,
            "initial": initial,
            "mask": mask,
            "patch_rows": (),
        }

    initial_snapshot = snapshot(initial, initial)
    final_snapshot = snapshot(initial, initial)
    common = {
        "cells": cells,
        "layout": "uniform",
        "initial_mass": 1.0,
        "probe_mass": 1.0,
        "final_mass": 1.0,
        "final_time": 1.0,
        "probe_phase_cycles": 0.37,
    }

    wrong_way = metrics(
        initial_snapshot,
        final_snapshot,
        snapshot(reverse, exact),
        **common,
    )
    transported = metrics(
        initial_snapshot,
        final_snapshot,
        snapshot(exact, exact),
        **common,
    )

    assert wrong_way["qualification"]["transport_probe_passed"] is False
    assert transported["qualification"]["transport_probe_passed"] is True
    assert (
        wrong_way["probe_against_reverse"]["errors"]["l2"]
        < wrong_way["probe_diagnostics"]["errors"]["l2"]
    )


def test_reverse_oracle_is_exact_on_an_asymmetric_composite_amr_mask():
    metrics = _compute_metrics()
    cell_averages = _support_scope()["sine_wave_cell_averages"]
    phase_cycles = 0.37
    coarse_mask = np.array([False, True, True, True])
    fine_mask = np.array([True, True, False, False, False, False, False, False])

    def snapshot(numeric_direction, reference_direction):
        result = {"patch_rows": ((1, 0, 1),)}
        for level, cells, mask in ((0, (4,), coarse_mask), (1, (8,), fine_mask)):
            initial, _ = cell_averages(cells, (1,), epsilon=0.1)
            exact, _ = cell_averages(
                cells,
                (1,),
                epsilon=0.1,
                displacement=(reference_direction * phase_cycles,),
            )
            numeric, _ = cell_averages(
                cells,
                (1,),
                epsilon=0.1,
                displacement=(numeric_direction * phase_cycles,),
            )
            result["initial_level_%d" % level] = initial
            result["exact_level_%d" % level] = exact
            result["numeric_level_%d" % level] = numeric
            result["mask_level_%d" % level] = mask
        return result

    initial_snapshot = snapshot(0.0, 0.0)
    final_snapshot = snapshot(0.0, 0.0)
    wrong_way = metrics(
        initial_snapshot,
        final_snapshot,
        snapshot(-1.0, 1.0),
        cells=(4,),
        layout="amr-frozen",
        initial_mass=1.0,
        probe_mass=1.0,
        final_mass=1.0,
        final_time=1.0,
        probe_phase_cycles=phase_cycles,
    )

    assert wrong_way["probe_against_reverse"]["errors"]["l2"] == pytest.approx(0.0)
    assert wrong_way["qualification"]["transport_probe_passed"] is False


def test_convergence_uses_the_non_periodic_probe_errors(tmp_path):
    scope = runpy.run_path(str(CASE / "plot_results.py"))
    plot_convergence = scope["_plot_convergence"]
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    execution = _test_execution_provenance()
    source_fingerprint = "a" * 64
    base = {
        "schema_version": SCHEMA_VERSION,
        "case": "periodic_sine_wave_advection",
        "dimension": 2,
        "mode": "diagonal",
        "wave_numbers": [1, 2],
        "velocity": [1.0, 1.0],
        "epsilon": 0.1,
        "probe_time": 0.37,
        "final_time": 1.0,
        "cycles": 1,
        "layout": "uniform",
        "subcycling": "synchronous",
        "block_size": 8,
        "patch_marker": None,
        "mpi": False,
        "mpi_ranks": 1,
        "timeline": {"frames": 9, "times": np.linspace(0.0, 1.0, 9).tolist()},
        "amr_diagnostics": _test_amr_diagnostics(),
        "provenance": {
            "pops_version": "test-version",
            "execution": execution,
            "source": {
                "repository_sha": "test-revision",
                "fingerprint": source_fingerprint,
            },
        },
        "result_identity_inputs": {
            "execution": execution,
            "source_fingerprint": source_fingerprint,
        },
        "resolution": [16, 16],
        "metrics": {
            "method": {"time": "SSPRK2", "cfl": 0.4},
            "qualification": {"passed": True},
            "errors": {"l1": 0.0, "l2": 0.0, "linf": 0.0},
            "probe_errors": {"l1": 0.1, "l2": 0.1, "linf": 0.1},
        },
    }
    other = json.loads(json.dumps(base))
    other["resolution"] = [32, 32]
    other["metrics"]["probe_errors"] = {"l1": 0.025, "l2": 0.025, "linf": 0.025}

    target = plot_convergence(plt, [base, other], tmp_path)

    assert target == tmp_path / "convergence.png"
    assert target.is_file()


@pytest.mark.parametrize(
    "difference",
    ("revision", "source", "backend", "velocity", "method"),
)
def test_convergence_rejects_mixed_scientific_or_execution_authorities(tmp_path, difference):
    plot_convergence = runpy.run_path(str(CASE / "plot_results.py"))["_plot_convergence"]
    execution = _test_execution_provenance()
    source_fingerprint = "a" * 64
    base = {
        "schema_version": SCHEMA_VERSION,
        "case": "periodic_sine_wave_advection",
        "dimension": 2,
        "mode": "diagonal",
        "wave_numbers": [1, 2],
        "velocity": [1.0, 1.0],
        "epsilon": 0.1,
        "probe_time": 0.37,
        "final_time": 1.0,
        "cycles": 1,
        "layout": "uniform",
        "subcycling": "synchronous",
        "block_size": 16,
        "patch_marker": None,
        "mpi": False,
        "mpi_ranks": 1,
        "timeline": {"frames": 9, "times": np.linspace(0.0, 1.0, 9).tolist()},
        "amr_diagnostics": _test_amr_diagnostics(),
        "provenance": {
            "pops_version": "test-version",
            "execution": execution,
            "source": {
                "repository_sha": "test-revision",
                "fingerprint": source_fingerprint,
            },
        },
        "result_identity_inputs": {
            "execution": execution,
            "source_fingerprint": source_fingerprint,
        },
        "resolution": [16, 16],
        "metrics": {
            "method": {"time": "SSPRK2", "cfl": 0.2},
            "qualification": {"run_integrity_passed": True},
            "probe_errors": {"l1": 0.1, "l2": 0.1, "linf": 0.1},
        },
    }
    other = json.loads(json.dumps(base))
    other["resolution"] = [32, 32]
    other["metrics"]["probe_errors"] = {"l1": 0.025, "l2": 0.025, "linf": 0.025}
    if difference == "revision":
        other["provenance"]["source"]["repository_sha"] = "other-revision"
    elif difference == "source":
        other["provenance"]["source"]["fingerprint"] = "b" * 64
        other["result_identity_inputs"]["source_fingerprint"] = "b" * 64
    elif difference == "backend":
        other["provenance"]["execution"]["runtime"]["kokkos_backend"] = "Cuda"
        other["result_identity_inputs"]["execution"]["runtime"]["kokkos_backend"] = "Cuda"
    elif difference == "velocity":
        other["velocity"] = [1.0, -1.0]
    else:
        other["metrics"]["method"]["cfl"] = 0.1

    assert plot_convergence(None, [base, other], tmp_path) is None


def test_case_support_keeps_the_declared_python_310_compatibility():
    source = SUPPORT.read_text(encoding="utf-8")
    assert "from datetime import UTC" not in source
    assert "datetime.now(timezone.utc)" in source
