"""Build and validate a per-run pops.verification.provenance.v1 document.

Plan §6.2: the monorepo SHA is the unique source revision. Extra digests
authenticate generated catalog, public headers, and the exact-rank native
leaf. This module does not run the solver.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import socket
import subprocess
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_provenance.v1.json"
SCHEMA_ID = "pops.verification.provenance.v1"
REPOSITORY = "wolf75222/PoPS"
ALLOWED_DIMENSIONS = (1, 2, 3)
MAX_NODES = 2
RUN_FIELDS = (
    "compiler",
    "build_type",
    "precision",
    "kokkos_execution_space",
    "mpi_enabled",
    "mpi_library",
    "mpi_thread_level_requested",
    "mpi_thread_level_provided",
    "hdf5_collective_enabled",
    "mpi_ranks",
    "omp_threads_per_rank",
    "gpus",
    "resolution",
    "block_size",
    "amr_total_levels",
    "refinement_ratio",
    "subcycling",
    "time_program",
    "cfl",
    "final_time",
)


class ProvenanceError(RuntimeError):
    pass


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def _validate_document(document: Mapping[str, Any]) -> None:
    try:
        _validator().validate(document)
    except ValidationError as exc:
        raise ProvenanceError(
            f"invalid provenance document: {exc.message}"
        ) from exc


def _require_non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"{name} must be a non-empty string")
    return value


def _repository_state(repo: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError("cannot read repository SHA") from exc
    if not sha:
        raise ProvenanceError("cannot read repository SHA")
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError("cannot read repository SHA") from exc
    return sha, bool(status.strip())


def _resolve_pops_version(override: str | None) -> str:
    try:
        import pops

        return str(pops.__version__)
    except Exception:
        pass
    if override is None:
        raise ProvenanceError(
            "pops is unavailable and pops_version was not provided"
        )
    return override


def _doctor_report_ok(report: Any) -> bool:
    if report is True:
        return True
    if not isinstance(report, Mapping) or not report:
        return False
    values = list(report.values())
    if all(
        isinstance(item, tuple) and item and item[0] is True for item in values
    ):
        return True
    return report.get("ok") is True


def _resolve_doctor_ok(override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    try:
        import pops

        report = pops.doctor(verbose=False)
    except Exception:
        return False
    return _doctor_report_ok(report)


def _public_catalog_digest() -> str | None:
    try:
        from pops.release import contract

        digest = contract()["component_catalog_sha256"]
    except Exception:
        return None
    if isinstance(digest, str) and digest:
        return digest
    return None


def _resolve_digest(name: str, supplied: str | None) -> str:
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied:
            raise ProvenanceError(f"empty digest: {name}")
        return supplied
    if name == "component_catalog_digest":
        public = _public_catalog_digest()
        if public is not None:
            return public
    raise ProvenanceError(f"empty digest: {name}")


def _run_fields(
    run: Mapping[str, Any] | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(extra) - set(RUN_FIELDS))
    if unknown:
        raise ProvenanceError(f"unknown run field: {unknown[0]}")
    merged: dict[str, Any] = {}
    if run is not None:
        merged.update(run)
    merged.update(extra)
    missing = [key for key in RUN_FIELDS if key not in merged]
    if missing:
        raise ProvenanceError(f"missing run field: {missing[0]}")
    copied: dict[str, Any] = {}
    for key in RUN_FIELDS:
        value = merged[key]
        if key in {"resolution", "block_size"}:
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def collect_provenance(
    case_id: str,
    *,
    pops_native_dim: int,
    dimension: int,
    nodes: int,
    run: Mapping[str, Any] | None = None,
    repo: str | Path | None = None,
    hostname: str | None = None,
    slurm_job_id: str | None = None,
    pops_version: str | None = None,
    doctor_ok: bool | None = None,
    component_catalog_digest: str | None = None,
    native_header_signature: str | None = None,
    native_variant_manifest_digest: str | None = None,
    **run_fields: Any,
) -> dict[str, Any]:
    """Return a schema-valid provenance document for one local or scheduled run."""
    _require_non_empty_string("case_id", case_id)
    if pops_native_dim not in ALLOWED_DIMENSIONS or dimension not in ALLOWED_DIMENSIONS:
        raise ProvenanceError("pops_native_dim and dimension must be 1, 2, or 3")
    if pops_native_dim != dimension:
        raise ProvenanceError("pops_native_dim and dimension must match")
    if not isinstance(nodes, int) or isinstance(nodes, bool) or not 1 <= nodes <= MAX_NODES:
        raise ProvenanceError("nodes must be 1 or 2")

    repo_path = REPO_ROOT if repo is None else Path(repo)
    repository_sha, repository_dirty = _repository_state(repo_path)
    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "case_id": case_id,
        "repository": REPOSITORY,
        "repository_sha": repository_sha,
        "repository_dirty": repository_dirty,
        "pops_version": _resolve_pops_version(pops_version),
        "component_catalog_digest": _resolve_digest(
            "component_catalog_digest", component_catalog_digest
        ),
        "native_header_signature": _resolve_digest(
            "native_header_signature", native_header_signature
        ),
        "native_variant_manifest_digest": _resolve_digest(
            "native_variant_manifest_digest", native_variant_manifest_digest
        ),
        "doctor_ok": _resolve_doctor_ok(doctor_ok),
        "date_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **_run_fields(run, run_fields),
        "pops_native_dim": pops_native_dim,
        "nodes": nodes,
        "hostname": socket.gethostname() if hostname is None else hostname,
        "slurm_job_id": (
            os.environ.get("SLURM_JOB_ID", "local")
            if slurm_job_id is None
            else slurm_job_id
        ),
        "dimension": dimension,
    }
    _validate_document(document)
    return document


def write_provenance(path: str | Path, document: Mapping[str, Any]) -> None:
    """Write ``document`` as JSON after schema validation. Refuse invalid input."""
    _validate_document(document)
    Path(path).write_text(json.dumps(dict(document), indent=2) + "\n", encoding="utf-8")
