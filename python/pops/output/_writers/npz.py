"""Exact serial NPZ scientific-output backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pops.output._writers.common import (
    PreparedOutputFile,
    ReopenedOutput,
    authenticate_manifest,
    json_text,
    manifest,
    serial_payload,
    temporary_path,
)
from pops.output.data import OutputRequest, OutputSnapshot, array_evidence


class NPZWriter:
    format = "npz"
    extension = ".npz"

    def prepare(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> PreparedOutputFile:
        if communicator is not None:
            raise ValueError("serial NPZ output cannot carry a communicator")
        if request.parallel:
            raise ValueError("NPZ has no collective writer; select HDF5(parallel=True)")
        import numpy as np

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("NPZ target must end in .npz")
        arrays, datasets, evidence = serial_payload(snapshot, request)
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
        return PreparedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.identity,
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
