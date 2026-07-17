"""Exact SERIAL, ROOT, and PER_RANK NPZ scientific-output backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pops.output._writers.common import (
    OutputWriterSession,
    ReopenedOutput,
    _StagedOutputFile,
    authenticate_manifest,
    json_text,
    manifest,
    piece_payload,
    temporary_path,
    validate_field_pieces,
    writer_execution_capability,
    writer_session_authority,
)
from pops.output.data import OutputRequest, OutputSnapshot, array_evidence


class NPZWriter:
    format = "npz"
    extension = ".npz"

    def __init__(self, mode: Any = None) -> None:
        from pops.output._consumer_contracts import ParallelMode

        if mode is None:
            mode = ParallelMode.SERIAL
        if type(mode) is not ParallelMode:
            raise TypeError("NPZWriter mode must be an exact ParallelMode")
        self._mode = mode

    def preflight(self, execution_context: Any) -> dict[str, Any]:
        from pops.output._consumer_contracts import ParallelMode

        if type(self._mode) is not ParallelMode:
            raise RuntimeError("NPZWriter preflight requires its resolved format mode")
        return writer_execution_capability(
            execution_context, self._mode, provider_id="pops.output.npz.v1")

    def prepare_session(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> OutputWriterSession:
        from pops.output._consumer_contracts import ParallelMode

        if request.parallel_mode is not self._mode:
            raise ValueError("NPZ writer mode differs from its resolved output request")
        if request.parallel_mode is ParallelMode.COLLECTIVE:
            raise ValueError("NPZ has no COLLECTIVE writer; select HDF5(COLLECTIVE)")
        if request.parallel_mode is ParallelMode.SERIAL and communicator is not None:
            raise ValueError("SERIAL NPZ writer session cannot carry a communicator")
        if request.parallel_mode is not ParallelMode.SERIAL and communicator is None:
            raise ValueError("distributed NPZ writer session requires its communicator")
        authority = writer_session_authority(self.format, request, target)

        def stage_file() -> _StagedOutputFile:
            return self._stage_file(snapshot, request, target)

        stage_callback = (
            stage_file
            if request.parallel_mode is not ParallelMode.ROOT or request.rank == 0
            else None
        )
        return OutputWriterSession(authority, stage_callback)

    def _stage_file(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
    ) -> _StagedOutputFile:
        from pops.output._consumer_contracts import ParallelMode

        import numpy as np

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("NPZ target must end in .npz")
        for field in snapshot.select(request):
            validate_field_pieces(
                field,
                snapshot.geometry(field.key),
                complete=request.parallel_mode is not ParallelMode.PER_RANK,
                rank=(None if request.parallel_mode is ParallelMode.ROOT else request.rank),
                size=request.size,
            )
        arrays, datasets, evidence = piece_payload(snapshot, request)
        output_manifest, identity = manifest(
            self.format, snapshot, request, evidence, datasets=datasets)
        temporary = temporary_path(target)
        payload = dict(arrays)
        payload["pops_output_manifest"] = np.asarray(json_text(output_manifest))
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        read_npz(temporary).require_selection(request)
        return _StagedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.publication_identity,
            verify=read_npz,
        )


def read_npz(path: Any) -> ReopenedOutput:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if "pops_output_manifest" not in files:
            raise ValueError("NPZ has no PoPS scientific output manifest")
        output_manifest, identity = authenticate_manifest(
            json.loads(str(data["pops_output_manifest"])), "npz")
        expected = set(output_manifest["arrays"]) | {"pops_output_manifest"}
        if files != expected:
            raise ValueError("NPZ keys differ from its exact output manifest")
        arrays = {
            name: np.asarray(data[name]).copy()
            for name in output_manifest["arrays"]
        }
    for name, evidence in output_manifest["arrays"].items():
        if array_evidence(arrays[name]) != evidence:
            raise ValueError("NPZ array %r failed content verification" % name)
    return ReopenedOutput(output_manifest, arrays, identity)


__all__ = ["NPZWriter", "read_npz"]
