"""Load a completed runner job or series directory as the only trust root.

The constructor performs every check. A hand-built dataclass, in-memory
mapping, or missing leaf path is never accepted.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.capabilities import (
    authenticate_installed_artifact,
    sha256_file,
)
from verification.pops_verify.evidence_contract import (
    EXTENSION_SLOTS,
    REQUIRED_JOB_FILES,
    REQUIRED_PAIR_FILES,
    read_digest_file,
)
from verification.pops_verify.oracle_producers import OracleProducerError, produce_oracle

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_METRICS = REPO_ROOT / "schemas" / "verification_metrics.v1.json"
SCHEMA_PROVENANCE = REPO_ROOT / "schemas" / "verification_provenance.v1.json"


class EvidenceError(ValueError):
    """Raised when on-disk evidence cannot be authenticated."""


def _hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_files(job_dir: Path, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not (job_dir / name).is_file()]
    if missing:
        raise EvidenceError(f"missing {label}: {missing}")


def _require_digest(path: Path, digest_path: Path, *, label: str) -> str:
    if not path.is_file():
        raise EvidenceError(f"missing {label} path")
    digest = read_digest_file(digest_path)
    actual = sha256_file(path)
    if digest != actual:
        raise EvidenceError(f"{label} digest mismatch")
    return actual


def _repository_sha(repo: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not sha:
        raise EvidenceError("cannot read repository SHA")
    return sha


def _validate_schema(document: Mapping[str, Any], schema_path: Path, *, label: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(document)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {path.name}") from exc
    if not isinstance(document, dict):
        raise EvidenceError(f"{path.name} must be a JSON object")
    return document


def _job_identity(resolved: Mapping[str, Any]) -> tuple[str, int, str, str]:
    job = resolved.get("job")
    if not isinstance(job, Mapping):
        raise EvidenceError("resolved_case.json missing job")
    case_id = str(job.get("case_id") or "")
    if not case_id:
        case = resolved.get("case")
        if isinstance(case, Mapping):
            case_id = str(case.get("id") or "")
    if not case_id:
        raise EvidenceError("resolved_case.json missing case_id")
    dimension = job.get("pops_native_dim")
    if dimension not in (1, 2, 3):
        raise EvidenceError("resolved_case.json missing dimension")
    space = str(job.get("execution_space") or "")
    mpi_mode = str(job.get("mpi_mode") or "")
    if not space or mpi_mode not in {"on", "off"}:
        raise EvidenceError("resolved_case.json missing space/MPI")
    return case_id, int(dimension), space, mpi_mode


def _load_job(job_dir: Path) -> dict[str, Any]:
    _require_files(job_dir, REQUIRED_JOB_FILES, label="job files")
    resolved = _load_json(job_dir / "resolved_case.json")
    resolved_digest = _require_digest(
        job_dir / "resolved_case.json",
        job_dir / "resolved_case.sha256",
        label="resolved",
    )
    provenance = _load_json(job_dir / "provenance.json")
    metrics = _load_json(job_dir / "metrics.json")
    _validate_schema(provenance, SCHEMA_PROVENANCE, label="provenance")
    _validate_schema(metrics, SCHEMA_METRICS, label="metrics")
    case_id, dimension, space, mpi_mode = _job_identity(resolved)
    if str(provenance.get("case_id")) != case_id:
        raise EvidenceError("provenance case_id does not match resolved case")
    if int(provenance.get("pops_native_dim") or 0) != dimension:
        raise EvidenceError("provenance dimension does not match resolved case")
    if int(provenance.get("dimension") or 0) != dimension:
        raise EvidenceError("provenance dimension does not match resolved case")
    if str(provenance.get("kokkos_execution_space")) != space:
        raise EvidenceError("provenance execution space does not match resolved case")
    mpi_enabled = bool(provenance.get("mpi_enabled"))
    if mpi_enabled != (mpi_mode == "on"):
        raise EvidenceError("provenance MPI does not match resolved case")
    if str(provenance.get("repository_sha")) != _repository_sha():
        raise EvidenceError("repository SHA mismatch")
    program_digest = _require_digest(
        job_dir / "program.bin",
        job_dir / "program.sha256",
        label="program",
    )
    result_digest = _require_digest(
        job_dir / "result.npy",
        job_dir / "result.sha256",
        label="result",
    )
    result = np.ascontiguousarray(np.load(job_dir / "result.npy"), dtype=np.float64)
    identity_doc = _load_json(job_dir / "native_artifact.json")
    leaf = Path(str(identity_doc.get("path") or ""))
    if not leaf.is_file():
        raise EvidenceError(f"native variant path is absent: {leaf}")
    leaf_digest = sha256_file(leaf)
    documented = str(identity_doc.get("sha256") or "")
    if not _hex64(documented) or leaf_digest != documented:
        raise EvidenceError("native artifact leaf digest mismatch")
    artifact_dim = identity_doc.get("dimension")
    if artifact_dim != dimension:
        raise EvidenceError("native artifact dimension does not match resolved case")
    variants_root = Path(str(identity_doc.get("variants_root") or ""))
    try:
        authenticated = authenticate_installed_artifact(
            dimension=int(artifact_dim),
            variants_root=variants_root,
            doctor_ok=False,
        )
    except Exception as exc:
        raise EvidenceError(f"leaf authentication failed: {exc}") from exc
    if sha256_file(authenticated.path) != leaf_digest:
        raise EvidenceError("rehashed leaf does not match native_artifact identity")
    if authenticated.sha256 != documented:
        raise EvidenceError("authenticated leaf digest mismatch")
    if authenticated.dimension != dimension:
        raise EvidenceError("authenticated leaf dimension mismatch")
    if str(provenance.get("native_header_signature")) != authenticated.native_header_signature:
        raise EvidenceError("header signature does not match installed checkout")
    if str(provenance.get("component_catalog_digest")) != authenticated.component_catalog_digest:
        raise EvidenceError("catalog digest does not match installed checkout")
    if (
        str(provenance.get("native_variant_manifest_digest"))
        != authenticated.native_variant_manifest_digest
    ):
        raise EvidenceError("variant manifest digest mismatch")
    pair_result = None
    pair_digest = ""
    if case_id == "TR-06":
        _require_files(job_dir, REQUIRED_PAIR_FILES, label="pair")
        pair_digest = _require_digest(
            job_dir / "pair_result.npy",
            job_dir / "pair_result.sha256",
            label="pair result",
        )
        _require_digest(
            job_dir / "pair_program.bin",
            job_dir / "pair_program.sha256",
            label="pair program",
        )
        pair_result = np.ascontiguousarray(
            np.load(job_dir / "pair_result.npy"), dtype=np.float64
        )
    try:
        oracle = produce_oracle(case_id, resolved, result, provenance)
    except OracleProducerError as exc:
        raise EvidenceError(str(exc)) from exc
    resolution = provenance.get("resolution") or [result.shape[-1]]
    n_cells = int(resolution[0]) if resolution else int(result.shape[-1])
    spacing = 1.0 / float(n_cells)
    linf = float(np.max(np.abs(result - oracle)))
    record = {
        "case_id": case_id,
        "result": result,
        "result_digest": result_digest,
        "sample_spacing": spacing,
        "leaf_sha256": leaf_digest,
        "native_header_signature": authenticated.native_header_signature,
        "native_variant_manifest_digest": authenticated.native_variant_manifest_digest,
        "component_catalog_digest": authenticated.component_catalog_digest,
        "program_digest": program_digest,
        "resolved_case_digest": resolved_digest,
        "pair_result": pair_result,
        "pair_result_digest": pair_digest,
        "provenance": provenance,
        "metrics": metrics,
        "resolved_case": resolved,
    }
    coupling_path = job_dir / EXTENSION_SLOTS["coupling"]
    if coupling_path.is_file():
        record["coupling_digest"] = sha256_file(coupling_path)
    mask_path = job_dir / EXTENSION_SLOTS["amr_mask"]
    if mask_path.is_file():
        record["amr_mask_digest"] = sha256_file(mask_path)
    record["derived_linf"] = linf
    return record


class EvidenceBundle:
    """Validated series or job directory. Constructor performs all checks."""

    __slots__ = (
        "path",
        "case_id",
        "records",
        "derived_linf",
        "derived_spacings",
        "_trusted",
    )

    def __new__(cls, path=None, *args, **kwargs):
        if cls is not EvidenceBundle:
            raise EvidenceError("EvidenceBundle cannot be subclass-forged")
        if not isinstance(path, (str, Path)):
            raise EvidenceError("EvidenceBundle requires a filesystem directory path")
        return object.__new__(cls)

    def __init__(self, path: str | Path):
        root = Path(path)
        if not root.is_dir():
            raise EvidenceError(f"missing evidence path: {root}")
        series_file = root / "series.json"
        if series_file.is_file():
            series = _load_json(series_file)
            case_id = str(series.get("case_id") or "")
            jobs = series.get("jobs")
            if not case_id or not isinstance(jobs, list) or not jobs:
                raise EvidenceError("series.json missing case_id/jobs")
            records = []
            for name in jobs:
                job_dir = root / str(name)
                if not job_dir.is_dir():
                    raise EvidenceError(f"missing job directory: {name}")
                record = _load_job(job_dir)
                if record["case_id"] != case_id:
                    raise EvidenceError("copied record case_id does not match series")
                records.append(record)
        else:
            records = [_load_job(root)]
            case_id = str(records[0]["case_id"])
        first = records[0]
        for record in records[1:]:
            if record["case_id"] != case_id:
                raise EvidenceError("series mixes case_id values")
            if record["leaf_sha256"] != first["leaf_sha256"]:
                raise EvidenceError("series mixes native leaf identities")
            space = record["provenance"].get("kokkos_execution_space")
            if space != first["provenance"].get("kokkos_execution_space"):
                raise EvidenceError("series mixes execution spaces")
            if record["provenance"].get("mpi_enabled") != first["provenance"].get(
                "mpi_enabled"
            ):
                raise EvidenceError("series mixes MPI modes")
            if record["provenance"].get("pops_native_dim") != first["provenance"].get(
                "pops_native_dim"
            ):
                raise EvidenceError("series mixes dimensions")
            if record["provenance"].get("repository_sha") != first["provenance"].get(
                "repository_sha"
            ):
                raise EvidenceError("series mixes repository SHA")
        self.path = root
        self.case_id = case_id
        self.records = tuple(records)
        self.derived_linf = tuple(float(item["derived_linf"]) for item in records)
        self.derived_spacings = tuple(float(item["sample_spacing"]) for item in records)
        self._trusted = True
