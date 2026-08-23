"""Static and pure-contract coverage for the macOS acquisition protocol."""

from __future__ import annotations

import json
import hashlib
import importlib.machinery
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
PROFILE = HARNESS / "profiling"
sys.path[:0] = [str(PROFILE), str(HARNESS)]

from collect_profiles import collect  # noqa: E402
from profile_contract import (  # noqa: E402
    CANONICAL,
    PROFILE_SCHEMA,
    ProfileContractError,
    canonical_plan,
    command_sha256,
    create_profile_complete_receipt,
    _inventory_sha256,
    _profile_inventory,
    external_profile_publication_path,
    exported_build_receipt,
    fresh_nonce,
    native_variant_receipt,
    profile_command,
    profile_figure_publication_path,
    sha256,
    source_manifest_receipt,
    tree_digest,
    verify_profile_complete_receipt,
    write_json_new,
)
from plot_profiles import main as render_profile_plots  # noqa: E402
from ready_go import ready_after_bind_warmup  # noqa: E402


BUILD_FINGERPRINT = "a" * 64


def _source_export(root: Path) -> tuple[Path, dict[str, object]]:
    source_root = root / "source-tree"
    required = (
        "benchmarks/performance/advection_sine/advection_sine.py",
        "benchmarks/performance/advection_sine/support.py",
        "benchmarks/performance/advection_sine/profiling/profile_contract.py",
        "benchmarks/performance/advection_sine/profiling/ready_go.py",
        "helpers/verification/sine_wave.py",
        "python/pops/__init__.py",
    )
    entries = []
    for index, relative in enumerate(required):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture-%d\n" % index, encoding="utf-8")
        entries.append(
            {
                "path": relative,
                "type": "file",
                "mode": 0o644,
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest = {
        "schema": "pops.performance.source-export.v1",
        "base_sha": "a" * 40,
        "source_dirty": False,
        "tree_sha256": digest,
        "archive_sha256": "b" * 64,
        "archive_format": "tar-pax",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = root / "source.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_root, {
        "base_sha": manifest["base_sha"],
        "source_dirty": manifest["source_dirty"],
        "tree_sha256": manifest["tree_sha256"],
        "archive_sha256": manifest["archive_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "file_count": manifest["file_count"],
    }


def _provenance(
    command: dict[str, object], source: dict[str, object], source_root: Path
) -> dict[str, object]:
    campaign_path = (HARNESS / "campaigns" / "strong_openmp.json").resolve()
    plan = canonical_plan(campaign_path)
    return {
        "campaign": {
            "path": str(campaign_path),
            "id": plan["campaign"]["id"],
            "point": plan["point"]["id"],
            "sha256": sha256(campaign_path),
        },
        "source": source,
        "python_package": {
            "path": "python/pops/__init__.py",
            "sha256": sha256(source_root / "python" / "pops" / "__init__.py"),
        },
        "artifact": {"semantic_identity": "artifact-token"},
        "program_artifact": {
            "artifact_identity": "artifact-token",
            "abi_key": "fixture-abi",
            "cache_key": "fixture-cache",
            "programs": [{"layout": "uniform", "sha256": "e" * 64, "size": 1}],
        },
        "native": {
            "manifest_sha256": "c" * 64,
            "path": "dim3/_pops.fixture",
            "sha256": "d" * 64,
            "dimension": 3,
            "version": "fixture",
            "abi_key": "fixture",
            "build_fingerprint": BUILD_FINGERPRINT,
            "has_mpi": False,
            "has_kokkos": True,
        },
        "build": {
            "filename": "build.receipt.json",
            "sha256": "f" * 64,
            "source_tree_sha256": source["tree_sha256"],
            "campaign": plan["campaign"]["id"],
            "native_sha256": "d" * 64,
            "native_build_fingerprint": BUILD_FINGERPRINT,
            "cmake_cache_sha256": "1" * 64,
        },
        "runtime": {
            "kokkos_backend": "OpenMP",
            "kokkos_concurrency": 8,
            "mpi_active": False,
            "mpi_ranks": 1,
        },
        "host": {"hostname": "test-host"},
        "command": command,
    }


def _write_leaf(root: Path, tool: str, repetition: int) -> None:
    leaf = root / tool / ("rep%02d" % repetition)
    leaf.mkdir(parents=True)
    nonce = "%064x" % (repetition + (0 if tool == "sample" else 16))
    output_dir = (leaf / "rank-output").resolve()
    source_root, source = _source_export(root)
    argv = profile_command(
        campaign_path=HARNESS / "campaigns" / "strong_openmp.json",
        python=Path(sys.executable),
        output_dir=output_dir,
        source_root=source_root,
    )
    command = {"argv": argv, "sha256": command_sha256(argv), "output_dir": str(output_dir)}
    ready = {
        "schema": PROFILE_SCHEMA,
        "phase": "ready_after_bind_warmup",
        "nonce": nonce,
        "pid": 1000 + repetition,
        "ready_unix_seconds": 10.0,
        "provenance": _provenance(command, source, source_root),
    }
    worker = {
        "schema": PROFILE_SCHEMA,
        "phase": "completed_public_lifecycle",
        "nonce": nonce,
        "pid": 1000 + repetition,
        "returncode": 0,
        "completed_unix_seconds": 20.0,
    }
    tool_receipt = {
        "schema": PROFILE_SCHEMA,
        "phase": "acquisition_complete",
        "tool": tool,
        "nonce": nonce,
        "started_unix_seconds": 15.0,
        "ended_unix_seconds": 25.0,
        "target_completed_unix_seconds": 20.0,
        "target_reaped_unix_seconds": 21.0,
        "profiler_reaped_unix_seconds": 25.0,
        "target_completed_during_acquisition": True,
        "attachment_proof": (
            "sample_header_before_go_after_stop" if tool == "sample" else "xctrace_notify_before_go"
        ),
    }
    command_receipt = {"schema": PROFILE_SCHEMA, "phase": "canonical_command", **command}
    profiler_exit = {
        "schema": PROFILE_SCHEMA,
        "phase": "profiler_exit",
        "tool": tool,
        "returncode": 0,
        "exited_unix_seconds": 25.0,
    }
    (leaf / "ready.json").write_text(json.dumps(ready), encoding="utf-8")
    (leaf / "command.json").write_text(json.dumps(command_receipt), encoding="utf-8")
    (leaf / "worker.receipt.json").write_text(json.dumps(worker), encoding="utf-8")
    (leaf / "tool.receipt.json").write_text(json.dumps(tool_receipt), encoding="utf-8")
    (leaf / "profiler.exit.json").write_text(json.dumps(profiler_exit), encoding="utf-8")
    output_dir.mkdir()
    (output_dir / "rank-00000.json").write_text(
        json.dumps(
            {
                "schema": "pops.performance.advection-sine.measurement.v3",
                "rank": 0,
                "campaign": ready["provenance"]["campaign"]["id"],
                "point": "t8",
                "timing": {"metric": "public_lifecycle_wall_seconds"},
                "validation": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    if tool == "sample":
        (leaf / "sample.txt").write_text(
            "Call graph:\n"
            "  10 public_parent (in PoPS)\n"
            "    7 public_frame (in PoPS)\n"
            "  3 other_frame (in libSystem)\n"
            "Binary Images:\n"
            "  999 ignored_outside_call_graph (in Ignored)\n",
            encoding="utf-8",
        )
    else:
        trace = leaf / "time-profiler.trace"
        trace.mkdir()
        (trace / "metadata.bin").write_bytes(b"trace data")
        (leaf / "toc.txt").write_text(
            '<trace-toc><table schema="time-profile" /></trace-toc>', encoding="utf-8"
        )


class MacosProfileContractTests(unittest.TestCase):
    def test_canonical_point_is_read_from_the_versioned_campaign(self) -> None:
        plan = canonical_plan(HARNESS / "campaigns" / "strong_openmp.json")
        self.assertEqual(plan["point"]["id"], "t8")
        self.assertEqual(plan["canonical"], CANONICAL)

    def test_nonce_is_long_unique_and_hex(self) -> None:
        first, second = fresh_nonce(), fresh_nonce()
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)
        self.assertTrue(all(character in "0123456789abcdef" for character in first))

    def test_profile_command_is_the_exact_canonical_t8_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            argv = profile_command(
                campaign_path=HARNESS / "campaigns" / "strong_openmp.json",
                python=Path(sys.executable),
                output_dir=Path(temporary) / "rank-output",
            )
            self.assertEqual(argv[1], "-B")
            self.assertEqual(argv[2], str(HARNESS / "advection_sine.py"))
            self.assertIn("--resolution=128,128,128", argv)
            self.assertIn("--block-size=32", argv)
            self.assertIn("--cfl=0.4", argv)
            self.assertIn("--steps=32", argv)
            self.assertIn("--warmups=1", argv)
            self.assertIn("--repetitions=5", argv)
            self.assertIn("--threads=8", argv)
            self.assertIn("--expected-ranks=1", argv)
            self.assertEqual(len(command_sha256(argv)), 64)

    def test_native_receipt_binds_selected_extension_to_variants_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            native_root = Path(temporary) / "_native"
            extension = native_root / "dim3" / ("_pops" + importlib.machinery.EXTENSION_SUFFIXES[0])
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"fixture-extension")
            manifest = {
                "schema_version": 2,
                "variants": [
                    {
                        "dimension": 3,
                        "path": extension.relative_to(native_root).as_posix(),
                        "sha256": sha256(extension),
                        "version": "fixture",
                        "abi_key": "fixture-abi",
                        "build_fingerprint": BUILD_FINGERPRINT,
                        "has_mpi": False,
                        "has_kokkos": True,
                    }
                ],
            }
            (native_root / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")

            class NativeFixture:
                __file__ = str(extension)
                __native_dimension__ = 3
                __version__ = "fixture"
                __has_mpi__ = False
                __has_kokkos__ = True
                __build_fingerprint__ = BUILD_FINGERPRINT

                @staticmethod
                def abi_key() -> str:
                    return "fixture-abi"

            receipt = native_variant_receipt(NativeFixture)
            self.assertEqual(receipt["sha256"], sha256(extension))
            self.assertEqual(receipt["build_fingerprint"], BUILD_FINGERPRINT)
            divergent = dict(manifest["variants"][0])
            divergent["dimension"] = 2
            divergent["path"] = "dim2/" + extension.name
            divergent["build_fingerprint"] = "b" * 64
            manifest["variants"].append(divergent)
            (native_root / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ProfileContractError, "disagree on their build fingerprint"
            ):
                native_variant_receipt(NativeFixture)
            manifest["variants"].pop()
            (native_root / "variants.json").write_text(json.dumps(manifest), encoding="utf-8")
            NativeFixture.__build_fingerprint__ = "b" * 64
            with self.assertRaisesRegex(ProfileContractError, "build fingerprint differs"):
                native_variant_receipt(NativeFixture)
            NativeFixture.__build_fingerprint__ = BUILD_FINGERPRINT
            extension.write_bytes(b"forged")
            with self.assertRaisesRegex(ProfileContractError, "bytes differ"):
                native_variant_receipt(NativeFixture)

    def test_build_receipt_fails_closed_without_matching_exported_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, source = _source_export(root)
            native = _provenance({"argv": ["fixture"], "sha256": "0" * 64, "output_dir": "x"}, source, source_root)["native"]
            receipt = {
                "schema": "pops.performance.advection-sine.build-receipt.v4",
                "source": {"tree_sha256": source["tree_sha256"]},
                "campaign": {"id": "romeo_x64_openmp_strong_3d"},
                "native_import": {
                    "extension": {
                        "sha256": native["sha256"],
                        "dimension": native["dimension"],
                        "build_fingerprint": native["build_fingerprint"],
                        "has_mpi": native["has_mpi"],
                        "has_kokkos": True,
                    }
                },
                "cmake": {"cache": {"sha256": "2" * 64}},
                "kokkos": {
                    "source_authority": {"kind": "installed-distribution"},
                    "libkokkoscore": {
                        "kind": "static-archive",
                        "path": "lib/libkokkoscore.a",
                        "sha256": "3" * 64,
                    },
                    "cmake_dir": {"path": "lib/cmake/Kokkos"},
                },
            }
            path = root / "build.receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            observed = exported_build_receipt(
                receipt_path=path,
                source=source,
                campaign_id="romeo_x64_openmp_strong_3d",
                native=native,
            )
            self.assertEqual(observed["native_sha256"], native["sha256"])
            receipt["kokkos"]["libkokkoscore"]["kind"] = "shared-library"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "cannot be proven"):
                exported_build_receipt(
                    receipt_path=path,
                    source=source,
                    campaign_id="romeo_x64_openmp_strong_3d",
                    native=native,
                )
            receipt["kokkos"]["libkokkoscore"]["kind"] = "static-archive"
            receipt["source"]["tree_sha256"] = "0" * 64
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "cannot be proven"):
                exported_build_receipt(
                    receipt_path=path,
                    source=source,
                    campaign_id="romeo_x64_openmp_strong_3d",
                    native=native,
                )

    def test_ready_requires_complete_authenticated_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "ready.json"
            output_dir = Path(temporary) / "rank-output"
            source_root, source = _source_export(Path(temporary))
            argv = profile_command(
                campaign_path=HARNESS / "campaigns" / "strong_openmp.json",
                python=Path(sys.executable),
                output_dir=output_dir,
                source_root=source_root,
            )
            command = {
                "argv": argv,
                "sha256": command_sha256(argv),
                "output_dir": str(output_dir.resolve()),
            }
            ready_after_bind_warmup(
                ready=ready, nonce="a" * 64, provenance=_provenance(command, source, source_root)
            )
            row = json.loads(ready.read_text(encoding="utf-8"))
            self.assertEqual(row["provenance"], _provenance(command, source, source_root))
            with self.assertRaises(ProfileContractError):
                ready_after_bind_warmup(
                    ready=Path(temporary) / "invalid.json",
                    nonce="b" * 64,
                    provenance={"campaign": {}},
                )

    def test_profile_contract_refuses_a_dirty_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, _source = _source_export(root)
            manifest_path = root / "source.manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_dirty"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "source_dirty=false"):
                source_manifest_receipt(manifest_path=manifest_path, source_root=source_root)

    def test_profile_complete_and_plot_refuse_a_reinventoried_dirty_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "profile"
            root.mkdir()
            for tool in ("sample", "xctrace"):
                for repetition in range(1, 6):
                    _write_leaf(root, tool, repetition)
            summary_path = root / "summary.json"
            summary = collect(root)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            create_profile_complete_receipt(root)

            summary["provenance"]["source"]["source_dirty"] = True
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            complete_path = root / "COMPLETE.json"
            receipt = json.loads(complete_path.read_text(encoding="utf-8"))
            receipt["summary"] = {"filename": "summary.json", "sha256": sha256(summary_path)}
            entries = _profile_inventory(root)
            receipt["evidence"] = {"sha256": _inventory_sha256(entries), "files": entries}
            complete_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(ProfileContractError, "summary is not bound"):
                verify_profile_complete_receipt(root)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "plot_profiles.py",
                    str(summary_path),
                    "--output",
                    str(Path(temporary) / "publication"),
                ],
            ):
                with self.assertRaisesRegex(ProfileContractError, "summary is not bound"):
                    render_profile_plots()

    def test_collector_requires_complete_acquisitions_and_unique_nonces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tool in ("sample", "xctrace"):
                for repetition in range(1, 6):
                    _write_leaf(root, tool, repetition)
            summary = collect(root)
            self.assertTrue(summary["scaling_excluded"])
            self.assertEqual(
                summary["sample_top15"],
                [
                    {"frame": "public_frame (in PoPS)", "weight": 35},
                    {"frame": "other_frame (in libSystem)", "weight": 15},
                ],
            )
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            complete = create_profile_complete_receipt(root)
            self.assertEqual(complete["runs"], {"sample": 5, "xctrace_time_profiler": 5})
            self.assertEqual(verify_profile_complete_receipt(root)["schema"], PROFILE_SCHEMA + ".complete.v1")
            (root / "summary.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "invalid profile summary"):
                verify_profile_complete_receipt(root)
            self.assertEqual(
                summary["sample_image_composition"],
                [{"image": "PoPS", "weight": 35}, {"image": "libSystem", "weight": 15}],
            )
            self.assertEqual(
                summary["sample_leaf_stacks"],
                [
                    {
                        "frames": ["public_parent (in PoPS)", "public_frame (in PoPS)"],
                        "weight": 35,
                    },
                    {"frames": ["other_frame (in libSystem)"], "weight": 15},
                ],
            )
            invalid_runtime = root / "sample" / "rep01" / "ready.json"
            row = json.loads(invalid_runtime.read_text(encoding="utf-8"))
            row["provenance"]["runtime"]["kokkos_concurrency"] = 7
            invalid_runtime.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "canonical OpenMP t8"):
                collect(root)
            row["provenance"]["runtime"]["kokkos_concurrency"] = 8
            invalid_runtime.write_text(json.dumps(row), encoding="utf-8")
            invalid_toc = root / "xctrace" / "rep01" / "toc.txt"
            invalid_toc.write_text(
                '<trace-toc><table schema="not-time-profile" /></trace-toc>', encoding="utf-8"
            )
            with self.assertRaisesRegex(ProfileContractError, "time-profile"):
                collect(root)
            invalid_toc.write_text(
                '<trace-toc><table schema="time-profile" /></trace-toc>', encoding="utf-8"
            )
            invalid_acquisition = root / "sample" / "rep01" / "tool.receipt.json"
            row = json.loads(invalid_acquisition.read_text(encoding="utf-8"))
            row["target_reaped_unix_seconds"] = 30.0
            invalid_acquisition.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "did not finish during acquisition"):
                collect(root)
            row["target_reaped_unix_seconds"] = 21.0
            invalid_acquisition.write_text(json.dumps(row), encoding="utf-8")
            row["profiler_reaped_unix_seconds"] = 19.0
            invalid_acquisition.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ProfileContractError, "did not finish during acquisition"):
                collect(root)
            row["profiler_reaped_unix_seconds"] = 25.0
            invalid_acquisition.write_text(json.dumps(row), encoding="utf-8")
            duplicate_leaf = root / "xctrace" / "rep05"
            duplicate_ready = json.loads(
                (duplicate_leaf / "ready.json").read_text(encoding="utf-8")
            )
            duplicate_worker = json.loads(
                (duplicate_leaf / "worker.receipt.json").read_text(encoding="utf-8")
            )
            duplicate_tool = json.loads(
                (duplicate_leaf / "tool.receipt.json").read_text(encoding="utf-8")
            )
            for row in (duplicate_ready, duplicate_worker, duplicate_tool):
                row["nonce"] = "%064x" % 1
            (duplicate_leaf / "ready.json").write_text(
                json.dumps(duplicate_ready), encoding="utf-8"
            )
            (duplicate_leaf / "worker.receipt.json").write_text(
                json.dumps(duplicate_worker), encoding="utf-8"
            )
            (duplicate_leaf / "tool.receipt.json").write_text(
                json.dumps(duplicate_tool), encoding="utf-8"
            )
            with self.assertRaisesRegex(ProfileContractError, "unique READY/GO nonce"):
                collect(root)

    def test_figure_publication_is_a_sibling_and_preserves_complete_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "profile"
            root.mkdir()
            for tool in ("sample", "xctrace"):
                for repetition in range(1, 6):
                    _write_leaf(root, tool, repetition)
            (root / "summary.json").write_text(json.dumps(collect(root)), encoding="utf-8")
            create_profile_complete_receipt(root)

            publication = profile_figure_publication_path(root)
            self.assertEqual(publication, Path(temporary) / "profile.figures")
            self.assertEqual(
                external_profile_publication_path(profile_root=root, publication_root=publication),
                publication.resolve(strict=False),
            )
            with mock.patch.dict(os.environ, {"MPLBACKEND": "Agg"}, clear=False), mock.patch.object(
                sys,
                "argv",
                ["plot_profiles.py", str(root / "summary.json"), "--output", str(publication)],
            ):
                self.assertEqual(render_profile_plots(), 0)

            self.assertTrue((publication / "plot_manifest.json").is_file())
            self.assertEqual(
                verify_profile_complete_receipt(root)["schema"], PROFILE_SCHEMA + ".complete.v1"
            )
            with self.assertRaisesRegex(ProfileContractError, "outside the sealed evidence root"):
                external_profile_publication_path(profile_root=root, publication_root=root / "figures")

    def test_trace_digest_is_recursive_and_sensitive_to_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "time-profiler.trace"
            (trace / "nested").mkdir(parents=True)
            (trace / "metadata.bin").write_bytes(b"first")
            (trace / "nested" / "detail.bin").write_bytes(b"second")
            first, count = tree_digest(trace)
            (trace / "nested" / "detail.bin").write_bytes(b"changed")
            second, changed_count = tree_digest(trace)
            self.assertEqual(count, 2)
            self.assertEqual(changed_count, 2)
            self.assertNotEqual(first, second)

    def test_no_clobber_receipt_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "receipt.json"
            write_json_new(target, {"ok": True})
            with self.assertRaises(ProfileContractError):
                write_json_new(target, {"ok": False})

    def test_runner_uses_real_start_signal_and_complete_process_options(self) -> None:
        source = (PROFILE / "run_macos_profile.sh").read_text(encoding="utf-8")
        for required in (
            "/usr/bin/xctrace",
            "/usr/bin/sample",
            "/usr/bin/notifyutil",
            "-mayDie",
            "-fullPaths",
            "--notify-tracing-started",
            "--quiet",
            "--no-prompt",
            "--time-limit",
            "target_reaped_unix_seconds",
            "write_canonical_command",
            "rank-output",
            "kill -STOP",
            "wait_sample_header",
            "kill -CONT",
            "prepare_export.py",
            "source.manifest.json",
            "source-tree",
            "native-build",
            '"${PYTHON}" -B',
            '-S "${EXPORTED_ROOT}"',
            "-DPOPS_BUILD_TESTS=OFF",
            "-DPOPS_USE_MPI=OFF",
            "-DPOPS_USE_HDF5=OFF",
            "CMAKE_HOME_DIRECTORY",
            "POPS_MACOS_PROFILE_KOKKOS_ROOT",
            "POPS_MACOS_PROFILE_BUILD_JOBS",
            '--parallel "${BUILD_JOBS}"',
            "cleanup_active_processes",
            "create_profile_complete_receipt",
            "profile_figure_publication_path",
            "verify_profile_complete_receipt",
            'mkdir -- "${OUTPUT}"',
            "for repetition in 1 2 3 4 5",
        ):
            self.assertIn(required, source)
        self.assertNotIn("PUBLIC_CASE_COMMAND", source)
        self.assertNotIn('"$@"', source)
        self.assertNotIn('--output "${OUTPUT}/figures"', source)
        self.assertNotIn("plot_scaling.py", source)
        self.assertNotIn("POPS_MACOS_PROFILE_BUILD_ROOT", source)


if __name__ == "__main__":
    unittest.main()
