"""JSON Schema for per-run verification provenance (plan §6.2)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_provenance.v1.json"

# Plan §6.2 example — field names and non-placeholder values verbatim.
# Angle-bracket placeholders are filled with non-empty strings; date_utc is already a date-time.
PLAN_SECTION_6_2_EXAMPLE = {
    "schema": "pops.verification.provenance.v1",
    "case_id": "CP-02",
    "repository": "wolf75222/PoPS",
    "repository_sha": "0123456789abcdef0123456789abcdef01234567",
    "repository_dirty": False,
    "pops_version": "1.0.0",
    "component_catalog_digest": (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "native_header_signature": (
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "native_variant_manifest_digest": (
        "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "doctor_ok": True,
    "date_utc": "2026-08-17T00:00:00Z",
    "compiler": "GCC 13.x",
    "build_type": "Release",
    "precision": "float64",
    "pops_native_dim": 2,
    "kokkos_execution_space": "OpenMP",
    "mpi_enabled": True,
    "mpi_library": "MPICH or OpenMPI, exact version",
    "mpi_thread_level_requested": "MPI_THREAD_MULTIPLE",
    "mpi_thread_level_provided": "MPI_THREAD_MULTIPLE",
    "hdf5_collective_enabled": True,
    "nodes": 1,
    "mpi_ranks": 4,
    "omp_threads_per_rank": 48,
    "gpus": 0,
    "hostname": "login1",
    "slurm_job_id": "12345",
    "dimension": 2,
    "resolution": [128, 128],
    "block_size": [32, 32],
    "amr_total_levels": 2,
    "refinement_ratio": 2,
    "subcycling": True,
    "time_program": "SSPRK2",
    "cfl": 0.4,
    "final_time": 1.0,
}

REQUIRED_TOP_LEVEL = (
    "schema",
    "case_id",
    "repository",
    "repository_sha",
    "repository_dirty",
    "pops_version",
    "component_catalog_digest",
    "native_header_signature",
    "native_variant_manifest_digest",
    "doctor_ok",
    "date_utc",
    "compiler",
    "build_type",
    "precision",
    "pops_native_dim",
    "kokkos_execution_space",
    "mpi_enabled",
    "mpi_library",
    "mpi_thread_level_requested",
    "mpi_thread_level_provided",
    "hdf5_collective_enabled",
    "nodes",
    "mpi_ranks",
    "omp_threads_per_rank",
    "gpus",
    "hostname",
    "slurm_job_id",
    "dimension",
    "resolution",
    "block_size",
    "amr_total_levels",
    "refinement_ratio",
    "subcycling",
    "time_program",
    "cfl",
    "final_time",
)

NON_EMPTY_DIGEST_FIELDS = (
    "repository_sha",
    "component_catalog_digest",
    "native_header_signature",
    "native_variant_manifest_digest",
)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def _plan_example() -> dict:
    return copy.deepcopy(PLAN_SECTION_6_2_EXAMPLE)


def _local_serial_run() -> dict:
    instance = _plan_example()
    instance["kokkos_execution_space"] = "Serial"
    instance["mpi_enabled"] = False
    instance["nodes"] = 1
    instance["gpus"] = 0
    instance["slurm_job_id"] = "local"
    return instance


def test_plan_section_6_2_example_is_valid():
    _validator().validate(_plan_example())


def test_plan_section_6_2_example_dimension_matches_pops_native_dim():
    instance = _plan_example()
    assert instance["dimension"] == instance["pops_native_dim"]
    _validator().validate(instance)


def test_local_one_node_serial_run_is_valid():
    instance = _local_serial_run()
    assert instance["nodes"] == 1
    assert instance["gpus"] == 0
    assert instance["mpi_enabled"] is False
    assert instance["slurm_job_id"] == "local"
    _validator().validate(instance)


def test_optional_doctor_report_and_device_inventory_are_accepted():
    instance = _plan_example()
    instance["doctor_report"] = {"ok": True, "notes": "archived doctor output"}
    instance["device_inventory"] = [{"name": "cpu", "count": 48}]
    _validator().validate(instance)
    instance["device_inventory"] = {"cpu": 48, "gpu": 0}
    _validator().validate(instance)


def test_rejects_wrong_schema_const():
    instance = _plan_example()
    instance["schema"] = "pops.verification.provenance.v0"
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("key", REQUIRED_TOP_LEVEL)
def test_rejects_missing_required_top_level_key(key):
    instance = _plan_example()
    del instance[key]
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_nodes_greater_than_two():
    instance = _plan_example()
    instance["nodes"] = 3
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_pops_native_dim_out_of_range():
    instance = _plan_example()
    instance["pops_native_dim"] = 4
    with pytest.raises(ValidationError):
        _validator().validate(instance)


@pytest.mark.parametrize("key", NON_EMPTY_DIGEST_FIELDS)
def test_rejects_empty_repository_sha_or_digest(key):
    instance = _plan_example()
    instance[key] = ""
    with pytest.raises(ValidationError):
        _validator().validate(instance)


def test_rejects_invalid_date_utc():
    instance = _plan_example()
    instance["date_utc"] = "not-a-date"
    with pytest.raises(ValidationError):
        _validator().validate(instance)
