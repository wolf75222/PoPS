"""Exact serial and collective HDF5 scientific-output backend."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from pops.output._writers.common import (
    PreparedOutputFile,
    ReopenedOutput,
    authenticate_manifest,
    json_text,
    manifest,
    selected_geometries,
    temporary_path,
)
from pops.output.data import OutputRequest, OutputSnapshot, array_evidence


def _require_h5py(parallel: bool) -> Any:
    try:
        h5py = import_module("h5py")
    except ImportError:
        raise RuntimeError("HDF5 output requires the optional h5py dependency") from None
    if parallel and not h5py.get_config().mpi:
        raise RuntimeError(
            "collective HDF5 requires h5py built with MPI; parallel=False is the serial route")
    return h5py


def _parallel_snapshot_data(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    target: Any,
    communicator: Any,
) -> tuple[dict[str, Any], tuple[Any, ...], Path, Any]:
    selected = ()
    target_path = None
    h5py = None
    try:
        if not request.parallel:
            raise ValueError("a resolved communicator is valid only for HDF5 parallel output")
        required = ("Get_rank", "bcast", "Barrier")
        if any(not callable(getattr(communicator, name, None)) for name in required):
            raise TypeError("collective HDF5 requires the resolved communicator")
        h5py = _require_h5py(True)
        target_path = Path(target)
        if target_path.suffix not in {".h5", ".hdf5"}:
            raise ValueError("HDF5 target must end in .h5 or .hdf5")
        data = snapshot.to_data(request)
        selected = snapshot.select(request)
        local_pieces = {
            field.key.identity.token: [piece.to_data() for piece in field.pieces]
            for field in selected
        }
        # Field ownership is rank-local; every other scientific fact must be identical before any
        # rank enters HDF5. Keeping the empty ``pieces`` member makes the compared value the exact
        # canonical snapshot schema rather than a parallel-only approximation of it.
        canonical = dict(data, fields=[dict(field, pieces=[]) for field in data["fields"]])
        envelope = {
            "snapshot": canonical,
            "pieces": local_pieces,
            "preflight": {
                "target": str(target_path.expanduser().resolve()),
                "h5py_version": str(getattr(h5py, "__version__", "unknown")),
                "hdf5_version": str(getattr(getattr(h5py, "version", None),
                                            "hdf5_version", "unknown")),
                "mpi": True,
            },
            "error": None,
        }
    except Exception as exc:
        # Every rank must still enter the collective so one malformed local snapshot cannot leave
        # its peers blocked in allgather. The detached error is diagnostic only; no output identity
        # is derived from it.
        envelope = {
            "snapshot": None,
            "pieces": None,
            "preflight": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    gathered = communicator.allgather(envelope)
    failures = [
        "rank %d: %s" % (rank, item["error"])
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if failures:
        raise ValueError(
            "collective HDF5 snapshot preparation failed across ranks: " + "; ".join(failures)
        )
    preflight = gathered[0]["preflight"]
    mismatched_preflight = [
        rank for rank, item in enumerate(gathered) if item["preflight"] != preflight
    ]
    if mismatched_preflight:
        raise ValueError(
            "collective HDF5 preflight differs across ranks: "
            + ", ".join(map(str, mismatched_preflight))
        )
    authority = gathered[0]["snapshot"]
    mismatched = [
        rank for rank, item in enumerate(gathered) if item["snapshot"] != authority
    ]
    if mismatched:
        raise ValueError(
            "collective HDF5 snapshot metadata differs across ranks: "
            + ", ".join(map(str, mismatched))
        )
    data = authority
    by_token = {
        field.key.identity.token: field
        for field in selected
    }
    expected_tokens = set(by_token)
    malformed = [
        rank
        for rank, item in enumerate(gathered)
        if set(item["pieces"]) != expected_tokens
    ]
    if malformed:
        raise ValueError(
            "collective HDF5 field ownership differs across ranks: "
            + ", ".join(map(str, malformed))
        )
    rebuilt = []
    for token in sorted(by_token):
        field = by_token[token]
        row = next(item for item in data["fields"] if item["key"] == field.key.to_data())
        pieces = [piece for rank in gathered for piece in rank["pieces"][token]]
        pieces.sort(key=lambda piece: (
            piece["lower"], piece["upper"], piece["array"]["content_sha256"]))
        active = []
        covered_cells = 0
        for piece in pieces:
            jlo, ilo = piece["lower"]
            jhi, ihi = piece["upper"]
            if (
                jlo < 0
                or ilo < 0
                or jhi <= jlo
                or ihi <= ilo
                or jhi > field.global_shape[0]
                or ihi > field.global_shape[1]
            ):
                raise ValueError("parallel field piece lies outside the global field")
            active = [other for other in active if other[1] > jlo]
            if any(not (ihi <= other[2] or other[3] <= ilo) for other in active):
                raise ValueError("parallel field pieces overlap across ranks")
            active.append((jlo, jhi, ilo, ihi))
            covered_cells += (jhi - jlo) * (ihi - ilo)
        if covered_cells != field.global_shape[0] * field.global_shape[1]:
            raise ValueError("parallel field pieces do not cover the global field")
        rebuilt.append(dict(row, pieces=pieces))
    data["fields"] = rebuilt
    if target_path is None or h5py is None:
        raise RuntimeError("collective HDF5 preflight completed without local authorities")
    if not isinstance(data, dict):
        raise RuntimeError("collective HDF5 snapshot authority is not canonical data")
    return data, selected, target_path, h5py


def _parallel_temporary_path(target: Path, communicator: Any) -> Path:
    """Create one shared temporary on rank zero and broadcast failures without deadlocking."""
    rank = int(communicator.Get_rank())
    envelope: dict[str, str | None] = {"path": None, "error": None}
    if rank == 0:
        try:
            envelope["path"] = str(temporary_path(target))
        except Exception as exc:
            envelope["error"] = "%s: %s" % (type(exc).__name__, exc)
    gathered = communicator.allgather(envelope)
    failures = [
        "rank %d: %s" % (owner, item["error"])
        for owner, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if failures:
        raise RuntimeError(
            "collective HDF5 temporary-file preparation failed: " + "; ".join(failures)
        )
    authority = gathered[0]["path"]
    if not authority or any(item["path"] is not None for item in gathered[1:]):
        raise RuntimeError("collective HDF5 temporary-file authority is malformed")
    return Path(authority)


class HDF5Writer:
    format = "hdf5"
    extension = ".h5"

    def prepare(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> PreparedOutputFile:
        if communicator is not None:
            # ``allgather`` is the one capability needed to make snapshot/request, h5py capability
            # and target-identity failures collective. The rest of the communicator protocol is
            # checked inside that first error envelope, before any rank enters HDF5.
            if not callable(getattr(communicator, "allgather", None)):
                raise TypeError("collective HDF5 requires the resolved communicator")
            snapshot_data, fields, target, h5py = _parallel_snapshot_data(
                snapshot, request, target, communicator
            )
        elif request.parallel:
            raise TypeError("collective HDF5 requires the resolved communicator")
        else:
            fields = snapshot.select(request)
            snapshot_data = snapshot.to_data(request)
            h5py = _require_h5py(False)
            target = Path(target)
            if target.suffix not in {".h5", ".hdf5"}:
                raise ValueError("HDF5 target must end in .h5 or .hdf5")
        arrays, datasets, evidence = {}, {"fields": {}, "geometries": {}}, {}
        for index, field in enumerate(fields):
            name = "fields/%04d/values" % index
            datasets["fields"][field.key.identity.token] = name
            if not request.parallel:
                arrays[name] = field.materialize()
        geometries = selected_geometries(snapshot, request, fields)
        for index, geometry in enumerate(sorted(geometries.values(), key=lambda item: item.key)):
            coverage = "geometry/%04d/coverage" % index
            valid = "geometry/%04d/valid_cells" % index
            volumes = "geometry/%04d/cell_volumes" % index
            arrays[coverage], arrays[valid], arrays[volumes] = (
                geometry.coverage, geometry.valid_cells, geometry.cell_volumes)
            datasets["geometries"]["%s#%d" % geometry.key] = {
                "coverage": coverage,
                "valid_cells": valid,
                "cell_volumes": volumes,
            }
        if request.parallel:
            for index, _field in enumerate(fields):
                name = "fields/%04d/values" % index
                global_row = snapshot_data["fields"][index]
                evidence[name] = {"pieces": global_row["pieces"]}
        evidence.update({name: array_evidence(value) for name, value in arrays.items()})
        output_manifest, identity = manifest(
            self.format,
            snapshot,
            request,
            evidence,
            snapshot_data=snapshot_data,
            datasets=datasets,
        )
        temporary = (
            _parallel_temporary_path(target, communicator)
            if request.parallel
            else temporary_path(target)
        )
        options = {"driver": "mpio", "comm": communicator} if request.parallel else {}
        rank = 0 if communicator is None else int(communicator.Get_rank())
        with h5py.File(temporary, "w", **options) as output:
            output.attrs["pops_output_manifest"] = json_text(output_manifest)
            for name, value in arrays.items():
                if request.parallel:
                    dataset = output.create_dataset(name, shape=value.shape, dtype=value.dtype)
                    if rank == 0:
                        dataset[...] = value
                else:
                    output.create_dataset(name, data=value, compression="gzip")
            for index, field in enumerate(fields):
                name = "fields/%04d/values" % index
                shape = (
                    ((len(field.component_names),) if field.component_names else ())
                    + field.global_shape
                )
                dataset = output.require_dataset(name, shape=shape, dtype=field.array_dtype)
                if request.parallel:
                    for piece in field.pieces:
                        jlo, ilo = piece.lower
                        jhi, ihi = piece.upper
                        dataset[..., jlo:jhi, ilo:ihi] = piece.values
            output.flush()
        if communicator is not None:
            communicator.Barrier()
        failure = None
        if communicator is None or communicator.Get_rank() == 0:
            try:
                read_hdf5(temporary).require_selection(request)
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            failure = communicator.bcast(failure, root=0)
        if failure is not None:
            raise RuntimeError("prepared HDF5 failed native verification: %s" % failure)
        if communicator is not None:
            communicator.Barrier()
        return PreparedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.identity,
            verify=read_hdf5,
            communicator=communicator,
        )


def read_hdf5(path: Any) -> ReopenedOutput:
    import numpy as np

    h5py = _require_h5py(False)
    with h5py.File(path, "r") as source:
        if "pops_output_manifest" not in source.attrs:
            raise ValueError("HDF5 has no PoPS scientific output manifest")
        output_manifest, identity = authenticate_manifest(
            json.loads(source.attrs["pops_output_manifest"]), "hdf5")
        arrays = {}
        for name, evidence in output_manifest["arrays"].items():
            if name not in source:
                raise ValueError("HDF5 lacks declared dataset %r" % name)
            value = np.asarray(source[name][...])
            arrays[name] = value
            if "pieces" in evidence:
                for piece in evidence["pieces"]:
                    jlo, ilo = piece["lower"]
                    jhi, ihi = piece["upper"]
                    if array_evidence(value[..., jlo:jhi, ilo:ihi]) != piece["array"]:
                        raise ValueError("HDF5 parallel piece failed verification")
            elif array_evidence(value) != evidence:
                raise ValueError("HDF5 dataset %r failed verification" % name)
        declared_roots = {name.split("/", 1)[0] for name in output_manifest["arrays"]}
        if set(source.keys()) != declared_roots:
            raise ValueError("HDF5 datasets differ from its exact manifest")
    return ReopenedOutput(output_manifest, arrays, identity)


__all__ = ["HDF5Writer", "read_hdf5"]
