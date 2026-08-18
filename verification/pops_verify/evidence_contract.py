"""On-disk evidence layout and the parent-runner emission contract.

Scientific pass is not minted here. ``emit_job_directory`` writes the files a
completed runner job must persist after the parent has authenticated the
installed exact-rank leaf. ``EvidenceBundle`` is the only loader/trust root.

Parent integration patch
------------------------
``scripts/run_verification.py`` currently writes ``resolved_case.json``,
``metrics.json``, and ``provenance.json`` only. It does **not** emit result
arrays, program bytes, or a native-artifact identity file. Until that changes,
owned case reports stay fail-closed.

Exact patch for ``_write_job_artifacts`` (and ``execute_jobs`` after a truthful
``run_native`` mapping):

1. Keep writing schema-valid ``resolved_case.json`` / ``metrics.json`` /
   ``provenance.json``.
2. If ``run_result`` is a mapping with ``result``, call
   ``emit_job_directory`` so the job directory also contains ``result.npy``,
   ``result.sha256``, ``program.bin``, ``program.sha256``,
   ``native_artifact.json``, and ``resolved_case.sha256``:
   - ``result``: ``numpy`` array from the payload (not a caller oracle);
   - ``program_bytes``: the compiled program image or serialized plan bytes
     already in the runner process — never ``repr(artifact)``;
   - ``native_artifact``: a JSON mapping ``{path, sha256, dimension,
     variants_root}`` copied from the parent-authenticated leaf (the runner
     already holds ``AuthenticatedArtifact``; serialize those four fields only);
   - ``pair_result`` / ``pair_program_bytes`` when the payload has
     ``pair_result`` (TR-06).
3. Write ``series.json`` at the case campaign directory listing each job
   subdirectory after all ranks of a resolution series complete.
4. Do not treat ``invoke_run_native`` success as a scientific pass. Analysis
   loads ``EvidenceBundle(series_dir)`` and only then may emit
   ``cases_passed=1``.

Do not invent compile-repr stand-ins or in-memory digest seals to skip this.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.capabilities import sha256_file
from verification.pops_verify.metrics import write_metrics
from verification.pops_verify.provenance import write_provenance

REQUIRED_JOB_FILES = (
    "resolved_case.json",
    "resolved_case.sha256",
    "provenance.json",
    "metrics.json",
    "result.npy",
    "result.sha256",
    "program.bin",
    "program.sha256",
    "native_artifact.json",
)

REQUIRED_PAIR_FILES = (
    "pair_result.npy",
    "pair_result.sha256",
    "pair_program.bin",
    "pair_program.sha256",
)

EXTENSION_SLOTS = {
    "coupling": "coupling.json",
    "amr_mask": "amr_mask.npy",
}

PARENT_INTEGRATION_PATCH = __doc__


class EvidenceContractError(ValueError):
    """Raised when a job directory cannot be written honestly."""


def write_digest_file(path: Path, digest: str) -> None:
    Path(path).write_text(str(digest).strip() + "\n", encoding="utf-8")


def read_digest_file(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _write_hashed_bytes(path: Path, data: bytes, digest_name: str) -> str:
    target = Path(path)
    target.write_bytes(data)
    digest = sha256_file(target)
    write_digest_file(target.with_name(digest_name), digest)
    return digest


def _write_hashed_array(path: Path, array: Any, digest_name: str) -> str:
    target = Path(path)
    stored = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    np.save(target, stored)
    digest = sha256_file(target)
    write_digest_file(target.with_name(digest_name), digest)
    return digest


def emit_job_directory(
    job_dir: str | Path,
    *,
    resolved_case: Mapping[str, Any],
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
    result: Any,
    program_bytes: bytes,
    native_artifact: Mapping[str, Any],
    pair_result: Any = None,
    pair_program_bytes: bytes | None = None,
    coupling: Mapping[str, Any] | None = None,
    amr_mask: Any = None,
) -> Path:
    """Write the contract job layout. Does not mint scientific pass."""
    if type(native_artifact).__name__ == "AuthenticatedArtifact":
        raise EvidenceContractError(
            "native_artifact must be a path identity mapping, not AuthenticatedArtifact"
        )
    if not isinstance(native_artifact, Mapping):
        raise EvidenceContractError("native_artifact must be a mapping")
    required = {"path", "sha256", "dimension", "variants_root"}
    missing = required - set(native_artifact)
    if missing:
        raise EvidenceContractError(f"native_artifact missing {sorted(missing)}")
    if not isinstance(program_bytes, (bytes, bytearray)):
        raise EvidenceContractError("program_bytes must be file bytes")
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "resolved_case.json"
    resolved_path.write_text(
        json.dumps(dict(resolved_case), indent=2) + "\n",
        encoding="utf-8",
    )
    write_digest_file(root / "resolved_case.sha256", sha256_file(resolved_path))
    write_provenance(root / "provenance.json", provenance)
    write_metrics(root / "metrics.json", metrics)
    _write_hashed_array(root / "result.npy", result, "result.sha256")
    _write_hashed_bytes(root / "program.bin", bytes(program_bytes), "program.sha256")
    artifact_path = root / "native_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "path": str(native_artifact["path"]),
                "sha256": str(native_artifact["sha256"]),
                "dimension": int(native_artifact["dimension"]),
                "variants_root": str(native_artifact["variants_root"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if pair_result is not None:
        if pair_program_bytes is None:
            raise EvidenceContractError("pair_program_bytes required with pair_result")
        _write_hashed_array(root / "pair_result.npy", pair_result, "pair_result.sha256")
        _write_hashed_bytes(
            root / "pair_program.bin", bytes(pair_program_bytes), "pair_program.sha256"
        )
    if coupling is not None:
        coupling_path = root / EXTENSION_SLOTS["coupling"]
        coupling_path.write_text(
            json.dumps(dict(coupling), indent=2) + "\n",
            encoding="utf-8",
        )
        write_digest_file(root / "coupling.sha256", sha256_file(coupling_path))
    if amr_mask is not None:
        _write_hashed_array(root / EXTENSION_SLOTS["amr_mask"], amr_mask, "amr_mask.sha256")
    return root
