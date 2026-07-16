"""Backend-independent scientific-output identity and publication transaction."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pops.identity import Identity, make_identity

from pops.output.data import OutputRequest, OutputSnapshot, array_evidence


OUTPUT_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def identity_from_token(token: Any, domain: str, where: str) -> Identity:
    try:
        result = Identity.from_token(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s has an invalid identity" % where) from exc
    if result.domain != domain:
        raise ValueError("%s must use the %r identity domain" % (where, domain))
    return result


def deterministic_target(
    directory: Any,
    prefix: Any,
    request: OutputRequest,
    snapshot: OutputSnapshot,
    extension: str,
) -> Path:
    """Return the sole deterministic, filesystem-bounded output filename.

    Human-readable prefixes are deliberately bounded, while the digest covers every full
    identity-bearing input.  Long consumer or clock identities therefore cannot exceed common
    ``NAME_MAX`` limits and cannot collide merely because their readable prefixes are equal.
    """
    root = Path(directory)
    clean_prefix = _SAFE_NAME.sub("-", str(prefix)).strip("-")
    clean_consumer = _SAFE_NAME.sub("-", request.consumer_id).strip("-")
    clean_clock = _SAFE_NAME.sub("-", snapshot.clock.clock_id).strip("-")
    if not clean_prefix or not clean_consumer or not clean_clock:
        raise ValueError("output filename parts must contain a safe non-empty token")
    if (not extension.startswith(".") or "/" in extension or "\\" in extension
            or len(extension.encode("utf-8")) > 32):
        raise ValueError("output extension must be a simple suffix")
    target_identity = make_identity("scientific-output-target", {
        "prefix": str(prefix),
        "consumer_id": request.consumer_id,
        "clock": snapshot.clock.to_data(),
        "request_identity": request.identity.token,
        "extension": extension,
    })
    name = "%s__%s__s%09d__%s%s" % (
        clean_prefix[:40],
        clean_consumer[:40],
        snapshot.clock.macro_step,
        target_identity.hexdigest,
        extension,
    )
    return root / name


def manifest(
    format_name: str,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    arrays: dict[str, Any],
    *,
    snapshot_data: dict[str, Any] | None = None,
    datasets: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Identity]:
    base = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "format": format_name,
        "snapshot": snapshot_data if snapshot_data is not None else snapshot.to_data(request),
        "datasets": datasets or {},
        "arrays": {name: arrays[name] for name in sorted(arrays)},
    }
    identity = make_identity("scientific-output", base)
    return dict(base, output_identity=identity.token), identity


def authenticate_manifest(
    value: Any,
    format_name: str,
) -> tuple[dict[str, Any], Identity]:
    if not isinstance(value, dict):
        raise TypeError("scientific output manifest must be a mapping")
    required = {
        "schema_version", "format", "snapshot", "datasets", "arrays", "output_identity",
    }
    if set(value) != required:
        raise ValueError("scientific output manifest keys are not exact")
    if value["schema_version"] != OUTPUT_SCHEMA_VERSION or value["format"] != format_name:
        raise ValueError("scientific output schema/format mismatch")
    supplied = identity_from_token(
        value["output_identity"], "scientific-output", "output_identity")
    base = {key: value[key] for key in required - {"output_identity"}}
    expected = make_identity("scientific-output", base)
    if supplied != expected:
        raise ValueError("scientific output manifest identity mismatch")
    return value, expected


@dataclass(frozen=True, slots=True)
class OutputPublicationReceipt:
    path: Path
    format: str
    output_identity: Identity
    selection_identity: Identity


class PreparedOutputFile:
    """Verified temporary scientific file, not yet attached to a consumer effect."""

    __slots__ = (
        "temporary", "target", "format", "output_identity", "selection_identity",
        "_verify", "_published", "_discarded", "_created_target", "_communicator",
    )

    def __init__(
        self,
        temporary: Any,
        target: Any,
        *,
        format: str,
        output_identity: Identity,
        selection_identity: Identity,
        verify: Callable[[Any], Any],
        communicator: Any = None,
    ) -> None:
        self.temporary, self.target = Path(temporary), Path(target)
        self.format = format
        self.output_identity, self.selection_identity = output_identity, selection_identity
        self._verify, self._communicator = verify, communicator
        self._published = self._discarded = False
        self._created_target = False

    def _rank(self) -> int:
        return 0 if self._communicator is None else int(self._communicator.Get_rank())

    def _barrier(self) -> None:
        if self._communicator is not None:
            self._communicator.Barrier()

    def publish(self) -> OutputPublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded output cannot be published")
        if self._published:
            return OutputPublicationReceipt(
                self.target, self.format, self.output_identity, self.selection_identity)
        self._barrier()
        failure = None
        if self._rank() == 0:
            try:
                self._verify(self.temporary)
                self.target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(self.temporary, self.target)
                    self._created_target = True
                except FileExistsError:
                    if hashlib.sha256(self.temporary.read_bytes()).digest() != hashlib.sha256(
                            self.target.read_bytes()).digest():
                        raise FileExistsError(
                            "scientific output collision at deterministic target %s" % self.target
                        ) from None
                self.temporary.unlink(missing_ok=True)
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if self._communicator is not None:
            failure = self._communicator.bcast(failure, root=0)
        if failure is not None:
            if self._communicator is None and failure.startswith("FileExistsError:"):
                raise FileExistsError(failure.split(": ", 1)[1])
            raise RuntimeError("collective output publication failed: %s" % failure)
        self._barrier()
        self._published = True
        return OutputPublicationReceipt(
            self.target, self.format, self.output_identity, self.selection_identity)

    def discard(self) -> None:
        if self._published or self._discarded:
            return
        self._barrier()
        if self._rank() == 0:
            self.temporary.unlink(missing_ok=True)
        self._barrier()
        self._discarded = True

    def rollback(self) -> None:
        """Compensate a staged or published output without deleting a pre-existing artifact."""
        if self._discarded:
            return
        self._barrier()
        if self._rank() == 0:
            self.temporary.unlink(missing_ok=True)
            if self._created_target:
                self.target.unlink(missing_ok=True)
        self._barrier()
        self._published = False
        self._discarded = True


@dataclass(frozen=True, slots=True)
class ReopenedOutput:
    manifest: dict[str, Any]
    arrays: dict[str, Any]
    output_identity: Identity

    def require_selection(self, request: OutputRequest) -> ReopenedOutput:
        recorded = self.manifest["snapshot"]["selection"]
        if recorded != request.to_data():
            raise ValueError("reopened output selection differs from the requested selection")
        return self


def temporary_path(target: Path, communicator: Any = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    rank = 0 if communicator is None else int(communicator.Get_rank())
    path = None
    if rank == 0:
        descriptor, name = tempfile.mkstemp(
            prefix=".%s." % target.name,
            suffix=".prepared",
            dir=str(target.parent),
        )
        os.close(descriptor)
        path = name
    if communicator is not None:
        path = communicator.bcast(path, root=0)
    if not isinstance(path, (str, os.PathLike)):
        raise RuntimeError("output temporary-path authority returned no filesystem path")
    return Path(path)


def serial_payload(
    snapshot: OutputSnapshot,
    request: OutputRequest,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    arrays, datasets = {}, {"fields": {}, "geometries": {}}
    fields = snapshot.select(request)
    for index, field in enumerate(fields):
        name = "field_%04d" % index
        arrays[name] = field.materialize()
        datasets["fields"][field.key.identity.token] = name
    geometries = selected_geometries(snapshot, request, fields)
    for index, geometry in enumerate(sorted(geometries.values(), key=lambda item: item.key)):
        coverage = "geometry_%04d_coverage" % index
        valid = "geometry_%04d_valid" % index
        volumes = "geometry_%04d_volumes" % index
        arrays[coverage], arrays[valid], arrays[volumes] = (
            geometry.coverage, geometry.valid_cells, geometry.cell_volumes)
        datasets["geometries"]["%s#%d" % geometry.key] = {
            "coverage": coverage,
            "valid_cells": valid,
            "cell_volumes": volumes,
        }
    evidence = {name: array_evidence(value) for name, value in arrays.items()}
    return arrays, datasets, evidence


def selected_geometries(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    fields: Any,
) -> dict[Any, Any]:
    geometries = {
        snapshot.geometry(field.key).key: snapshot.geometry(field.key)
        for field in fields
    }
    diagnostic_layouts = {item.layout_identity.token for item in request.diagnostics}
    geometries.update({
        item.key: item
        for item in snapshot.geometries
        if item.layout_identity.token in diagnostic_layouts
    })
    return geometries


__all__ = [
    "OUTPUT_SCHEMA_VERSION", "OutputPublicationReceipt", "PreparedOutputFile", "ReopenedOutput",
    "deterministic_target",
]
