"""Load a completed runner job or series directory as the only trust root.

The constructor performs every check against the installed leaf authority.
A hand-built dataclass, private variants root, symlink, or in-memory
mapping is never accepted.
"""
from __future__ import annotations

import json
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from verification.pops_verify.capabilities import (
    authenticate_installed_artifact,
    resolve_variants_root,
    sha256_file,
)
from verification.pops_verify.evidence_contract import (
    EXTENSION_SLOTS,
    REQUIRED_JOB_FILES,
    REQUIRED_PAIR_FILES,
    read_digest_file,
)
from verification.pops_verify.oracle_producers import (
    OracleProducerError,
    produce_oracle,
    produce_paired_oracle,
    verify_committed_producers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_METRICS = REPO_ROOT / "schemas" / "verification_metrics.v1.json"
SCHEMA_PROVENANCE = REPO_ROOT / "schemas" / "verification_provenance.v1.json"
SAFE_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_RESULT_NDIM = 3
MAX_RESULT_ELEMS = 4_000_000
JOB_PINNED_FILES = (
    "resolved_case.json",
    "resolved_case.sha256",
    "provenance.json",
    "metrics.json",
    "result.npy",
    "result.sha256",
    "program.bin",
    "program.sha256",
    "native_artifact.json",
    "producer_manifest.json",
    "producer_manifest.sha256",
)


class EvidenceError(ValueError):
    """Raised when on-disk evidence cannot be authenticated."""


def sample_spacing(
    case_id: str,
    provenance: Mapping[str, Any],
    resolved: Mapping[str, Any],
    result: Any,
) -> float:
    """Return Δx, or TM-01 Δt from the trusted resolved job — never 1/N."""
    if str(case_id) == "TM-01":
        job = resolved.get("job") if isinstance(resolved.get("job"), Mapping) else {}
        if job.get("dt") is None:
            raise EvidenceError("TM-01 requires dt in resolved job")
        dt = float(job["dt"])
        if not np.isfinite(dt) or dt <= 0.0:
            raise EvidenceError("TM-01 dt must be a positive finite value")
        n_cells = int(job.get("min_resolution") or (provenance.get("resolution") or [64])[0])
        if n_cells != 64:
            raise EvidenceError("TM-01 uses fixed N=64")
        return dt
    resolution = provenance.get("resolution") or [int(np.asarray(result).shape[-1])]
    return 1.0 / float(resolution[0])


def _hex64(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


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


def _reject_symlink(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"missing {label} path") from exc
    if stat.S_ISLNK(info.st_mode):
        raise EvidenceError(f"symlinked {label} is not accepted")


def _safe_read_bytes(path: Path, *, label: str) -> bytes:
    _reject_symlink(path, label=label)
    before = path.lstat()
    data = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise EvidenceError(f"{label} changed during read")
    if stat.S_ISLNK(after.st_mode):
        raise EvidenceError(f"symlinked {label} is not accepted")
    return data


def _safe_sha256(path: Path, *, label: str) -> str:
    import hashlib

    digest = hashlib.sha256(_safe_read_bytes(path, label=label)).hexdigest()
    return digest


def _pinned_under(root: Path, name: str, *, label: str) -> Path:
    if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
        raise EvidenceError(f"{label} escapes job root")
    path = root / name
    _reject_symlink(path, label=label)
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise EvidenceError(f"{label} escapes job root") from exc
    return path


def _safe_job_name(name: Any) -> str:
    if not isinstance(name, str) or not SAFE_JOB_NAME.fullmatch(name):
        raise EvidenceError("job name must be one safe relative component")
    if name.startswith(".") or ".." in name:
        raise EvidenceError("job name escape")
    return name


def _resolve_job_dir(series_root: Path, name: str) -> Path:
    safe = _safe_job_name(name)
    _reject_symlink(series_root, label="series")
    job_dir = series_root / safe
    _reject_symlink(job_dir, label="job")
    resolved = job_dir.resolve()
    root_resolved = series_root.resolve()
    if resolved.parent != root_resolved:
        raise EvidenceError("job name escape")
    if not job_dir.is_dir():
        raise EvidenceError(f"missing job directory: {safe}")
    return job_dir


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = _safe_read_bytes(path, label=label)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {path.name}") from exc
    if not isinstance(document, dict):
        raise EvidenceError(f"{path.name} must be a JSON object")
    return document


def _require_digest(path: Path, digest_path: Path, *, label: str) -> str:
    digest = read_digest_file(digest_path)
    actual = _safe_sha256(path, label=label)
    if digest != actual:
        raise EvidenceError(f"{label} digest mismatch")
    return actual


def _load_numeric_npy(path: Path, *, label: str) -> np.ndarray:
    _reject_symlink(path, label=label)
    before = path.lstat()
    try:
        with path.open("rb") as handle:
            loaded = np.load(handle, allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise EvidenceError(f"{label} pickle/dtype/shape is not accepted") from exc
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise EvidenceError(f"{label} changed during read")
    if not isinstance(loaded, np.ndarray):
        raise EvidenceError(f"{label} must be a numeric ndarray")
    if loaded.ndim == 0 or loaded.ndim > MAX_RESULT_NDIM:
        raise EvidenceError(f"{label} shape is not accepted")
    if loaded.size > MAX_RESULT_ELEMS:
        raise EvidenceError(f"{label} shape is not accepted")
    if loaded.size == 0:
        raise EvidenceError(f"{label} shape is not accepted")
    if not np.issubdtype(loaded.dtype, np.number) or np.issubdtype(loaded.dtype, np.complexfloating):
        raise EvidenceError(f"{label} dtype is not accepted")
    array = np.ascontiguousarray(loaded, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise EvidenceError(f"{label} contains non-finite values")
    array.flags.writeable = False
    return array


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


def _authenticate_installed_leaf(identity_doc: Mapping[str, Any], dimension: int):
    documented = str(identity_doc.get("sha256") or "")
    if not _hex64(documented):
        raise EvidenceError("native artifact leaf digest mismatch")
    raw_path = Path(str(identity_doc.get("path") or ""))
    if not raw_path.is_file():
        raise EvidenceError(f"native variant path is absent: {raw_path}")
    _reject_symlink(raw_path, label="leaf")
    try:
        installed_root = resolve_variants_root(None)
        authenticated = authenticate_installed_artifact(
            dimension=dimension,
            variants_root=installed_root,
            doctor_ok=False,
        )
    except Exception as exc:
        raise EvidenceError(f"installed leaf authentication failed: {exc}") from exc
    leaf = raw_path.resolve()
    installed = Path(authenticated.path).resolve()
    if leaf != installed:
        raise EvidenceError("native_artifact.path is not the installed leaf")
    live = sha256_file(installed)
    if live != authenticated.sha256 or live != documented:
        raise EvidenceError("live leaf digest mismatch")
    return authenticated, live


def _analyze_tr06_pair(
    job_dir: Path,
    *,
    case_id: str,
    resolved: Mapping[str, Any],
    provenance: Mapping[str, Any],
    result: np.ndarray,
    program_digest: str,
    leaf_sha256: str,
) -> tuple[np.ndarray, str, str, float]:
    for name in REQUIRED_PAIR_FILES:
        _pinned_under(job_dir, name, label=name)
    pair_digest = _require_digest(
        job_dir / "pair_result.npy",
        job_dir / "pair_result.sha256",
        label="pair result",
    )
    pair_program_digest = _require_digest(
        job_dir / "pair_program.bin",
        job_dir / "pair_program.sha256",
        label="pair program",
    )
    if pair_program_digest == program_digest:
        raise EvidenceError("TR-06 pair program identity must be distinct")
    if pair_program_digest == leaf_sha256:
        raise EvidenceError("pair program digest must not equal leaf digest")
    pair_result = _load_numeric_npy(job_dir / "pair_result.npy", label="pair result")
    if pair_result.shape != result.shape:
        raise EvidenceError("TR-06 pair result shape mismatch")
    try:
        pair_oracle = produce_paired_oracle(case_id, resolved, pair_result, provenance)
    except OracleProducerError as exc:
        raise EvidenceError(str(exc)) from exc
    pair_linf = float(np.max(np.abs(pair_result - pair_oracle)))
    bound = 10.0 * (float(np.max(np.abs(pair_oracle))) + 1.0)
    if not np.isfinite(pair_linf) or pair_linf > bound:
        raise EvidenceError("TR-06 pair result does not match the independent paired oracle")
    return pair_result, pair_digest, pair_program_digest, pair_linf


def _load_job(job_dir: Path, *, expected_case_id: str | None = None) -> dict[str, Any]:
    _reject_symlink(job_dir, label="job")
    for name in JOB_PINNED_FILES:
        _pinned_under(job_dir, name, label=name)
    missing = [name for name in REQUIRED_JOB_FILES if not (job_dir / name).is_file()]
    if missing:
        raise EvidenceError(f"missing job files: {missing}")
    resolved_path = job_dir / "resolved_case.json"
    resolved = _load_json(resolved_path, label="resolved")
    resolved_digest = _require_digest(
        resolved_path, job_dir / "resolved_case.sha256", label="resolved"
    )
    provenance = _load_json(job_dir / "provenance.json", label="provenance")
    metrics = _load_json(job_dir / "metrics.json", label="metrics")
    _validate_schema(provenance, SCHEMA_PROVENANCE, label="provenance")
    _validate_schema(metrics, SCHEMA_METRICS, label="metrics")
    case_id, dimension, space, mpi_mode = _job_identity(resolved)
    if expected_case_id is not None and case_id != expected_case_id:
        raise EvidenceError("copied record case_id does not match series")
    if str(provenance.get("case_id")) != case_id:
        raise EvidenceError("provenance case_id does not match resolved case")
    if int(provenance.get("pops_native_dim") or 0) != dimension:
        raise EvidenceError("provenance dimension does not match resolved case")
    if int(provenance.get("dimension") or 0) != dimension:
        raise EvidenceError("provenance dimension does not match resolved case")
    if str(provenance.get("kokkos_execution_space")) != space:
        raise EvidenceError("provenance execution space does not match resolved case")
    if bool(provenance.get("mpi_enabled")) != (mpi_mode == "on"):
        raise EvidenceError("provenance MPI does not match resolved case")
    if str(provenance.get("repository_sha")) != _repository_sha():
        raise EvidenceError("repository SHA mismatch")
    _require_digest(
        job_dir / "producer_manifest.json",
        job_dir / "producer_manifest.sha256",
        label="producer manifest",
    )
    stored_producers = _load_json(job_dir / "producer_manifest.json", label="producer")
    try:
        live_producers = verify_committed_producers(case_id)
    except OracleProducerError as exc:
        raise EvidenceError(str(exc)) from exc
    required = {
        "verification/pops_verify/cell_averages.py",
        "verification/pops_verify/oracle_producers.py",
    }
    if required - set(stored_producers):
        raise EvidenceError("producer manifest missing oracle-affecting sources")
    if set(stored_producers) != set(live_producers):
        raise EvidenceError("producer manifest keys do not match HEAD producers")
    for rel, live_row in live_producers.items():
        stored_row = stored_producers.get(rel)
        if not isinstance(stored_row, Mapping):
            raise EvidenceError(f"producer manifest missing {rel}")
        stored_sha = str(stored_row.get("sha256") or "")
        stored_head = str(stored_row.get("head_sha256") or "")
        stored_blob = str(stored_row.get("git_blob") or "")
        live_sha = str(live_row.get("sha256") or "")
        live_head = str(live_row.get("head_sha256") or "")
        live_blob = str(live_row.get("git_blob") or "")
        if (
            stored_sha != stored_head
            or stored_sha != live_sha
            or stored_head != live_head
            or live_sha != live_head
            or stored_blob != live_blob
        ):
            raise EvidenceError(f"producer {rel} does not match HEAD")
    identity_doc = _load_json(job_dir / "native_artifact.json", label="native artifact")
    documented_leaf = str(identity_doc.get("sha256") or "")
    program_digest = _require_digest(
        job_dir / "program.bin", job_dir / "program.sha256", label="program"
    )
    if program_digest == documented_leaf:
        raise EvidenceError("program digest must not equal leaf digest")
    result_digest = _require_digest(
        job_dir / "result.npy", job_dir / "result.sha256", label="result"
    )
    result = _load_numeric_npy(job_dir / "result.npy", label="result")
    pair_result = None
    pair_digest = ""
    pair_program_digest = ""
    pair_linf = None
    if case_id == "TR-06":
        pair_result, pair_digest, pair_program_digest, pair_linf = _analyze_tr06_pair(
            job_dir,
            case_id=case_id,
            resolved=resolved,
            provenance=provenance,
            result=result,
            program_digest=program_digest,
            leaf_sha256=documented_leaf,
        )
    try:
        oracle = produce_oracle(case_id, resolved, result, provenance)
    except OracleProducerError as exc:
        raise EvidenceError(str(exc)) from exc
    authenticated, leaf_digest = _authenticate_installed_leaf(identity_doc, dimension)
    if program_digest == leaf_digest:
        raise EvidenceError("program digest must not equal leaf digest")
    if pair_program_digest and pair_program_digest == leaf_digest:
        raise EvidenceError("pair program digest must not equal leaf digest")
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
    spacing = sample_spacing(case_id, provenance, resolved, result)
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
        "pair_program_digest": pair_program_digest,
        "derived_linf": linf,
        "derived_pair_linf": pair_linf,
        "provenance": MappingProxyType(dict(provenance)),
        "metrics": MappingProxyType(dict(metrics)),
        "resolved_case": MappingProxyType(dict(resolved)),
    }
    coupling_path = job_dir / EXTENSION_SLOTS["coupling"]
    if coupling_path.is_file():
        _pinned_under(job_dir, EXTENSION_SLOTS["coupling"], label="coupling")
        record["coupling_digest"] = _safe_sha256(coupling_path, label="coupling")
    mask_name = EXTENSION_SLOTS["amr_mask"]
    if (job_dir / mask_name).is_file():
        _pinned_under(job_dir, mask_name, label="amr_mask")
        record["amr_mask_digest"] = _safe_sha256(job_dir / mask_name, label="amr_mask")
    return dict(record)


class EvidenceBundle:
    """Validated series directory. Constructor performs all checks."""

    __slots__ = (
        "_path",
        "_case_id",
        "_records",
        "_derived_linf",
        "_derived_spacings",
        "_derived_pair_linf",
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
        _reject_symlink(root, label="series")
        series_file = root / "series.json"
        if series_file.is_file():
            _reject_symlink(series_file, label="series.json")
            series = _load_json(series_file, label="series")
            case_id = str(series.get("case_id") or "")
            jobs = series.get("jobs")
            if not case_id or not isinstance(jobs, list) or not jobs:
                raise EvidenceError("series.json missing case_id/jobs")
            records = []
            for name in jobs:
                record = _load_job(_resolve_job_dir(root, name), expected_case_id=case_id)
                if record["case_id"] != case_id:
                    raise EvidenceError("copied record case_id does not match series")
                records.append(MappingProxyType(record))
        else:
            loaded = _load_job(root)
            case_id = str(loaded["case_id"])
            records = [MappingProxyType(loaded)]
        first = records[0]
        for record in records[1:]:
            if record["case_id"] != case_id:
                raise EvidenceError("series mixes case_id values")
            if record["leaf_sha256"] != first["leaf_sha256"]:
                raise EvidenceError("series mixes native leaf identities")
            if record["provenance"].get("kokkos_execution_space") != first["provenance"].get(
                "kokkos_execution_space"
            ):
                raise EvidenceError("series mixes execution spaces")
            if record["provenance"].get("mpi_enabled") != first["provenance"].get("mpi_enabled"):
                raise EvidenceError("series mixes MPI modes")
            if record["provenance"].get("pops_native_dim") != first["provenance"].get(
                "pops_native_dim"
            ):
                raise EvidenceError("series mixes dimensions")
            if record["provenance"].get("repository_sha") != first["provenance"].get(
                "repository_sha"
            ):
                raise EvidenceError("series mixes repository SHA")
        object.__setattr__(self, "_path", root)
        object.__setattr__(self, "_case_id", case_id)
        object.__setattr__(self, "_records", tuple(records))
        object.__setattr__(
            self, "_derived_linf", tuple(float(item["derived_linf"]) for item in records)
        )
        object.__setattr__(
            self,
            "_derived_spacings",
            tuple(float(item["sample_spacing"]) for item in records),
        )
        object.__setattr__(
            self,
            "_derived_pair_linf",
            tuple(
                float(item["derived_pair_linf"])
                for item in records
                if item.get("derived_pair_linf") is not None
            ),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise EvidenceError("EvidenceBundle state is immutable")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return self._records

    @property
    def derived_linf(self) -> tuple[float, ...]:
        return self._derived_linf

    @property
    def derived_spacings(self) -> tuple[float, ...]:
        return self._derived_spacings

    @property
    def derived_pair_linf(self) -> tuple[float, ...]:
        return self._derived_pair_linf
