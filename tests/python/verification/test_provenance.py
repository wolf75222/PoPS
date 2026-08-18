"""Collector/writer for per-run verification provenance (plan §6.2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.provenance import (
    ProvenanceError,
    collect_provenance,
    write_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_provenance.v1.json"

DIGESTS = {
    "component_catalog_digest": (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "native_header_signature": (
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
    "native_variant_manifest_digest": (
        "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ),
}

LOCAL_RUN = {
    "compiler": "GCC 13.x",
    "build_type": "Release",
    "precision": "float64",
    "kokkos_execution_space": "Serial",
    "mpi_enabled": False,
    "mpi_library": "none",
    "mpi_thread_level_requested": "MPI_THREAD_SINGLE",
    "mpi_thread_level_provided": "MPI_THREAD_SINGLE",
    "hdf5_collective_enabled": False,
    "mpi_ranks": 1,
    "omp_threads_per_rank": 1,
    "gpus": 0,
    "resolution": [32],
    "block_size": [16],
    "amr_total_levels": 1,
    "refinement_ratio": 2,
    "subcycling": False,
    "time_program": "SSPRK2",
    "cfl": 0.4,
    "final_time": 1.0,
}


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def _collect(**overrides):
    kwargs = {
        "case_id": "CP-02",
        "pops_native_dim": 1,
        "dimension": 1,
        "nodes": 1,
        "run": dict(LOCAL_RUN),
        "doctor_ok": True,
        "repo": REPO_ROOT,
        **DIGESTS,
    }
    kwargs.update(overrides)
    return collect_provenance(**kwargs)


def test_fully_specified_local_document_validates(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    document = _collect()
    assert document["schema"] == "pops.verification.provenance.v1"
    assert document["repository"] == "wolf75222/PoPS"
    assert document["case_id"] == "CP-02"
    assert document["nodes"] == 1
    assert document["slurm_job_id"] == "local"
    assert document["pops_native_dim"] == document["dimension"] == 1
    assert document["component_catalog_digest"] == DIGESTS["component_catalog_digest"]
    assert document["native_header_signature"] == DIGESTS["native_header_signature"]
    assert (
        document["native_variant_manifest_digest"]
        == DIGESTS["native_variant_manifest_digest"]
    )
    assert document["repository_sha"]
    assert isinstance(document["repository_dirty"], bool)
    assert document["date_utc"].endswith("Z")
    _validator().validate(document)


def test_nodes_greater_than_two_raises():
    with pytest.raises(ProvenanceError, match="nodes"):
        _collect(nodes=3)


def test_empty_case_id_or_digest_raises():
    with pytest.raises(ProvenanceError, match="case_id"):
        _collect(case_id="")
    with pytest.raises(ProvenanceError, match="digest"):
        _collect(component_catalog_digest="")
    with pytest.raises(ProvenanceError, match="digest"):
        _collect(native_header_signature="")
    with pytest.raises(ProvenanceError, match="digest"):
        _collect(native_variant_manifest_digest="")


def test_mismatched_native_dim_and_dimension_raises():
    with pytest.raises(ProvenanceError, match="dimension"):
        _collect(pops_native_dim=1, dimension=2)


def test_write_provenance_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    document = _collect()
    path = tmp_path / "provenance.json"
    write_provenance(path, document)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    loaded = json.loads(text)
    assert loaded == document
    _validator().validate(loaded)


def test_write_provenance_refuses_invalid_document(tmp_path):
    path = tmp_path / "provenance.json"
    with pytest.raises(ProvenanceError):
        write_provenance(path, {"schema": "pops.verification.provenance.v1"})
    assert not path.exists()


def test_missing_git_sha_raises(tmp_path):
    with pytest.raises(ProvenanceError, match="(?i)sha"):
        _collect(repo=tmp_path)
