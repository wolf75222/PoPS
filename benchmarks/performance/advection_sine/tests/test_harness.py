"""Pure contract tests; they never import or execute a PoPS numerical runtime."""

from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from collect_results import _expected_dt, _read_rank_rows, _validate_point  # noqa: E402
from common import (  # noqa: E402
    CANONICAL_CAMPAIGN_FILENAMES,
    CampaignError,
    load_campaign,
    slurm_arguments,
    validate_canonical_campaign_inventory,
)
from plot_scaling import _authenticated_summary_root  # noqa: E402
from prepare_export import (  # noqa: E402
    ExportError,
    _tree_digest,
    create_complete_receipt,
    verify_complete_receipt,
)
from support import write_rank_measurement  # noqa: E402


class PublicPythonHarnessTests(unittest.TestCase):
    def test_all_campaigns_are_v2_public_lifecycle_contracts(self) -> None:
        for path in sorted((HARNESS / "campaigns").glob("*.json")):
            campaign = load_campaign(path)
            self.assertEqual(campaign["schema"], "pops.performance.advection-sine.campaign.v2")
            self.assertGreater(campaign["cfl"], 0.0)
            self.assertLess(campaign["cfl"], 1.0)

    def test_canonical_inventory_rejects_missing_extra_or_reduced_campaigns(self) -> None:
        campaigns = HARNESS / "campaigns"
        self.assertEqual({path.name for path in campaigns.glob("*.json")}, CANONICAL_CAMPAIGN_FILENAMES)
        validate_canonical_campaign_inventory(campaigns)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "campaigns"
            shutil.copytree(campaigns, copied)
            (copied / "serial_reference.json").unlink()
            with self.assertRaisesRegex(CampaignError, "missing"):
                validate_canonical_campaign_inventory(copied)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "campaigns"
            shutil.copytree(campaigns, copied)
            payload = json.loads((copied / "strong_openmp.json").read_text(encoding="utf-8"))
            payload["steps"] = 1
            (copied / "strong_openmp.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CampaignError, "reviewed full matrix"):
                validate_canonical_campaign_inventory(copied)

    def test_scaling_plotter_refuses_an_orphan_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary_path = Path(temporary) / "report" / "summary.json"
            summary_path.parent.mkdir()
            summary = {"campaign": "fixture"}
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "COMPLETE"):
                _authenticated_summary_root(summary_path, summary)

    def test_publications_generate_analysis_inside_the_hashed_staging_directory(self) -> None:
        scaling = (HARNESS / "plot_scaling.py").read_text(encoding="utf-8")
        profile = (HARNESS / "profiling" / "plot_profiles.py").read_text(encoding="utf-8")
        for source in (scaling, profile):
            self.assertIn('staging / "ANALYSIS.md"', source)
            self.assertIn("_write_publication_manifest(staging", source)
        self.assertIn("uniforme et synchrone", scaling)

    def test_walltime_escalation_is_short_only_and_does_not_mutate_campaign(self) -> None:
        campaign = load_campaign(HARNESS / "campaigns" / "serial_reference.json")
        arguments = slurm_arguments(
            campaign, partition_override="short", time_override="02:00:00"
        )
        self.assertIn("--partition=short", arguments)
        self.assertIn("--time=02:00:00", arguments)
        self.assertEqual(campaign["slurm"]["partition"], "instant")
        with self.assertRaisesRegex(CampaignError, "only submit-time escalation"):
            slurm_arguments(campaign, partition_override="instant")

    def _serial_row(self, campaign: dict) -> dict:
        point = campaign["points"][0]
        errors = {"l1": 0.0, "l2": 0.0, "linf": 0.0}
        final = {"l1": 0.01, "l2": 0.02, "linf": 0.03}
        stationary = {"l1": 0.2, "l2": 0.2, "linf": 0.2}
        return {
            "schema": "pops.performance.advection-sine.measurement.v3",
            "campaign": campaign["id"],
            "point": point["id"],
            "route": campaign["route"],
            "rank": 0,
            "metadata": {
                "execution_space": "Serial",
                "execution_concurrency": 1,
                "mpi_ranks": 1,
                "gpu_device_ordinal": -1,
                "gpu_uuid": "",
                "gpu_uuid_method": "none",
                "gpu_uuid_diagnostic": "",
                "omp_proc_bind": "spread",
                "omp_places": "cores",
                "omp_dynamic": "false",
            },
            "program_artifact": {
                "artifact_identity": "artifact-token",
                "abi_key": "fixture-abi",
                "cache_key": "fixture-cache",
                "programs": [{"layout": "uniform", "sha256": "a" * 64, "size": 1}],
            },
            "problem": {
                "dimension": campaign["dimension"],
                "resolution": point["resolution"],
                "mode": campaign["mode"],
                "layout": "uniform",
                "block_size": campaign["block_size"],
                "cfl": campaign["cfl"],
                "dt": _expected_dt(campaign, point),
                "steps": campaign["steps"],
            },
            "resources": {
                "nodes": point["nodes"],
                "ranks": point["ranks"],
                "threads_per_rank": point["threads"],
            },
            "local_boxes": [
                {"lower": [0] * campaign["dimension"], "upper_exclusive": point["resolution"]}
            ],
            "timing": {
                "metric": "public_lifecycle_wall_seconds",
                "samples": [1.0] * campaign["repetitions"],
            },
            "validation": {
                "timed": False,
                "passed": True,
                "initial_exact_errors": errors,
                "final_exact_errors": final,
                "stationary_initial_errors": stationary,
                "final_to_stationary_l2_ratio": 0.1,
                "native_integral": 1.0,
                "host_integral": 1.0,
                "initial_native_integral": 1.0,
                "initial_host_integral": 1.0,
                "mass_drift": 0.0,
                "nonfinite_final_cells": 0,
            },
        }

    def test_collector_requires_rank_owned_exact_cover(self) -> None:
        campaign = load_campaign(HARNESS / "campaigns" / "serial_reference.json")
        row = self._serial_row(campaign)
        normalized = _validate_point(campaign, campaign["points"][0], [row])
        self.assertEqual(normalized["median_seconds"], 1.0)
        row["local_boxes"] = []
        with self.assertRaisesRegex(CampaignError, "no rank-owned"):
            _validate_point(campaign, campaign["points"][0], [row])

    def test_collector_rejects_weak_dissipation_and_gpu_identity_on_cpu(self) -> None:
        campaign = load_campaign(HARNESS / "campaigns" / "serial_reference.json")
        row = self._serial_row(campaign)
        row["validation"]["final_to_stationary_l2_ratio"] = 0.932
        with self.assertRaisesRegex(CampaignError, "final exact-error guards"):
            _validate_point(campaign, campaign["points"][0], [row])
        row = self._serial_row(campaign)
        row["metadata"]["gpu_device_ordinal"] = 0
        with self.assertRaisesRegex(CampaignError, "GPU identity"):
            _validate_point(campaign, campaign["points"][0], [row])

    def test_collector_rejects_missing_or_mutated_program_binary_authority(self) -> None:
        campaign = load_campaign(HARNESS / "campaigns" / "serial_reference.json")
        row = self._serial_row(campaign)
        row.pop("program_artifact")
        with self.assertRaisesRegex(CampaignError, "program artifact"):
            _validate_point(campaign, campaign["points"][0], [row])
        row = self._serial_row(campaign)
        row["program_artifact"]["programs"][0]["sha256"] = "z" * 64
        with self.assertRaisesRegex(CampaignError, "Program binary authority"):
            _validate_point(campaign, campaign["points"][0], [row])

    def test_rank_publication_refuses_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            write_rank_measurement(target, 0, {"schema": "test"})
            with self.assertRaises(FileExistsError):
                write_rank_measurement(target, 0, {"schema": "test"})

    def test_collector_binds_rank_filename_to_payload_rank(self) -> None:
        campaign = load_campaign(HARNESS / "campaigns" / "serial_reference.json")
        point = campaign["points"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / point["id"]
            output.mkdir()
            row = self._serial_row(campaign)
            row["rank"] = 0
            (output / "rank-00001.json").write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(CampaignError, "filename rank"):
                _read_rank_rows(root, point)

    def test_romeo_mpi_routes_explicitly_disable_hdf5(self) -> None:
        for name in ("x64cpu.sbatch", "armgpu.sbatch"):
            wrapper = (HARNESS / "slurm" / name).read_text(encoding="utf-8")
            self.assertIn("-DPOPS_USE_HDF5=OFF", wrapper)
            self.assertNotIn('-DPOPS_USE_HDF5="${MPI_ENABLED}"', wrapper)

    def test_romeo_benchmark_builds_do_not_configure_the_test_tree(self) -> None:
        for name in ("x64cpu.sbatch", "armgpu.sbatch"):
            wrapper = (HARNESS / "slurm" / name).read_text(encoding="utf-8")
            self.assertEqual(wrapper.count("-DPOPS_BUILD_TESTS=OFF"), 1)

    def test_romeo_tree_verification_cannot_create_untracked_bytecode(self) -> None:
        """The verifier must not mutate the authenticated extracted source tree."""
        invocation = (
            '"${JOB_PYTHON}" -B "${HARNESS}/prepare_export.py" verify-tree'
        )
        for name in ("x64cpu.sbatch", "armgpu.sbatch"):
            wrapper = (HARNESS / "slurm" / name).read_text(encoding="utf-8")
            self.assertIn(invocation, wrapper)

    def test_romeo_job_python_dependency_contract_is_fail_closed(self) -> None:
        """Batch jobs must retain their activated NumPy/pybind11 path."""
        platform_contracts = {
            "x64cpu": (
                "X64",
                "x64cpu",
                "/rkvb73h",
                "/j4cl5xe",
                "/apps/2025/manual_install/spack-x64cpu-1.0.x/share/spack/setup-env.sh",
            ),
            "armgpu": (
                "ARMGPU",
                "armgpu",
                "/qwvf7fx",
                "/oshtzly",
                "/apps/2025/manual_install/spack-armgpu-1.0.x/share/spack/setup-env.sh",
            ),
        }
        for platform, (
            suffix,
            submitter,
            numpy_pin,
            pybind11_pin,
            spack_setup,
        ) in platform_contracts.items():
            wrapper_path = HARNESS / "slurm" / f"{platform}.sbatch"
            wrapper = wrapper_path.read_text(encoding="utf-8")
            submitted = (HARNESS / "slurm" / f"submit_{submitter}.sh").read_text(
                encoding="utf-8"
            )
            syntax = subprocess.run(
                ["bash", "-n", str(wrapper_path)], capture_output=True, text=True, check=False
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            self.assertIn(f"POPS_PYTHON_NUMPY_{suffix}_SPEC", wrapper)
            self.assertIn(f"POPS_PYBIND11_{suffix}_SPEC", wrapper)
            self.assertIn(numpy_pin, wrapper)
            self.assertIn(pybind11_pin, wrapper)
            self.assertIn(f"POPS_PYTHON_DEPENDENCY_ACTIVATION_{suffix}", wrapper)
            self.assertIn(f"POPS_PYBIND11_DIR_{suffix}", wrapper)
            self.assertIn("import numpy", wrapper)
            self.assertIn("import pybind11", wrapper)
            self.assertIn("python-dependencies.json", wrapper)
            self.assertIn("environment-provenance.json", wrapper)
            self.assertIn('"schema": "pops.environment-provenance.v1"', wrapper)
            self.assertNotIn("env |", wrapper)
            self.assertNotIn("${USER}", wrapper)
            self.assertNotIn("${HOME}", wrapper)
            self.assertIn("POPS_PERF_SUBMIT_USER", wrapper)
            self.assertIn("POPS_PERF_SUBMIT_HOME", wrapper)
            self.assertIn("${PERF_USER}", wrapper)
            self.assertIn("${PERF_HOME}", wrapper)
            self.assertIn('PATH="/usr/bin:/bin"', wrapper)
            self.assertIn(f'ROMEO_SPACK_SETUP="{spack_setup}"', wrapper)
            self.assertIn('source "${ROMEO_SPACK_SETUP}"', wrapper)
            self.assertNotIn("romeo_load_", wrapper)
            if platform == "armgpu":
                self.assertIn('ROMEO_MODULES_SETUP="/etc/profile.d/modules.sh"', wrapper)
                self.assertIn('source "${ROMEO_MODULES_SETUP}"', wrapper)
            self.assertIn("PYBIND11_DIR=\"$(resolve_romeo_pybind11_dir)\"", wrapper)
            self.assertIn("-Dpybind11_DIR=\"${PYBIND11_DIR}\"", wrapper)
            self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", wrapper)
            self.assertIn('PYTHON_DEPENDENCY_PATH="${PYTHONPATH:-}"', wrapper)
            self.assertIn(
                'PYTHONPATH="${WORK_ROOT}/build/python:${WORK_ROOT}/source'
                '${PYTHON_DEPENDENCY_PATH:+:${PYTHON_DEPENDENCY_PATH}}"',
                wrapper,
            )
            self.assertIn(
                'PYTHONPATH="${WORK_ROOT}/source'
                '${PYTHON_DEPENDENCY_PATH:+:${PYTHON_DEPENDENCY_PATH}}"',
                wrapper,
            )
            self.assertIn('[[ "${JOB_PYTHON}" != /* ]]', wrapper)
            self.assertIn('[[ "${python_path}" != /* ]]', submitted)
            for unsafe_mode in ("ALL", "NONE", "NIL"):
                self.assertNotIn(f"--export={unsafe_mode}", submitted)
            self.assertIn('SBATCH_EXPORT=""', submitted)
            self.assertIn('SBATCH_EXPORT="${name}=${value}"', submitted)
            self.assertIn('SBATCH_EXPORT+=",${name}=${value}"', submitted)
            self.assertIn('"--export=${SBATCH_EXPORT}"', submitted)
            self.assertIn("append_sbatch_export", submitted)
            self.assertIn("POPS_PERF_SUBMIT_USER POPS_PERF_SUBMIT_HOME", submitted)

    def test_mpi_build_receipt_distinguishes_slurm_and_openmpi_launchers(self) -> None:
        source = (HARNESS / "prepare_export.py").read_text(encoding="utf-8")
        self.assertIn('_command_receipt("srun", "SLURM srun launcher")', source)
        self.assertIn('_command_receipt("mpirun", "OpenMPI launcher")', source)

    def test_build_receipt_authenticates_native_build_fingerprint(self) -> None:
        exporter = (HARNESS / "prepare_export.py").read_text(encoding="utf-8")
        collector = (HARNESS / "collect_results.py").read_text(encoding="utf-8")
        self.assertIn("native.__build_fingerprint__", exporter)
        self.assertIn('extension.get("build_fingerprint")', collector)

    def test_complete_receipt_seals_raw_and_collected_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            report = root / "report"
            raw.mkdir()
            report.mkdir()
            campaign_bytes = b'{"id":"serial_reference"}\n'
            tree = "a" * 64
            files = [
                {
                    "path": "benchmarks/performance/advection_sine/campaigns/serial_reference.json",
                    "type": "file",
                    "mode": 0o644,
                    "size": len(campaign_bytes),
                    "sha256": hashlib.sha256(campaign_bytes).hexdigest(),
                }
            ]
            manifest = {
                "schema": "pops.performance.source-export.v1",
                "base_sha": "b" * 40,
                "source_dirty": False,
                "tree_sha256": _tree_digest(files),
                "archive_sha256": "c" * 64,
                "archive_format": "tar-pax",
                "file_count": len(files),
                "files": files,
            }
            tree = manifest["tree_sha256"]
            (raw / "source.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            build = {
                "schema": "pops.performance.advection-sine.build-receipt.v2",
                "source": {"tree_sha256": tree},
                "campaign": {"id": "serial_reference"},
            }
            (raw / "build.receipt.json").write_text(json.dumps(build), encoding="utf-8")
            summary = {
                "campaign": "serial_reference",
                "source_manifest": manifest,
                "build_receipt": build,
            }
            (report / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (report / "measurements.csv").write_text("point\nreference\n", encoding="utf-8")
            create_complete_receipt(
                raw,
                report,
                "serial_reference",
                "42",
                tree,
                root / "COMPLETE.json",
            )
            self.assertEqual(verify_complete_receipt(root)["campaign"], "serial_reference")
            (report / "measurements.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ExportError, "inventory differs"):
                verify_complete_receipt(root)


if __name__ == "__main__":
    unittest.main()
