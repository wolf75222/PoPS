"""Exact SERIAL, ROOT, COLLECTIVE, and PER_RANK HDF5 scientific-output backend."""
from __future__ import annotations

import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from pops.output._writers.common import (
    OutputWriterSession,
    ReopenedOutput,
    _StagingAuthority,
    _StagedOutputFile,
    _cleanup_staging_authority,
    authenticate_manifest,
    json_text,
    manifest,
    selected_geometries,
    temporary_path,
    validate_field_pieces,
    writer_execution_capability,
    writer_session_authority,
)
from pops.output.data import OutputRequest, OutputSnapshot, array_evidence


def _require_h5py(parallel: bool = False) -> Any:
    if parallel:
        raise RuntimeError(
            "collective HDF5 is implemented only by the compiled C++ HDF5 C backend"
        )
    try:
        h5py = import_module("h5py")
    except ImportError:
        raise RuntimeError("HDF5 output requires the optional h5py dependency") from None
    return h5py


def _require_native_parallel_hdf5() -> tuple[Any, dict[str, Any]]:
    from pops import _pops

    available = getattr(_pops, "__has_parallel_hdf5__", None)
    capability = getattr(_pops, "_parallel_hdf5_capability", None)
    write = getattr(_pops, "_write_parallel_hdf5", None)
    if available is not True or not callable(capability) or not callable(write):
        raise RuntimeError(
            "collective HDF5 requires a module built with POPS_USE_MPI=ON, "
            "POPS_USE_HDF5=ON, and a parallel HDF5 C library"
        )
    report = capability()
    required = {"available", "hdf5_version", "reason", "communicator", "implementation"}
    if type(report) is not dict or set(report) != required or report["available"] is not True:
        raise RuntimeError("native collective HDF5 capability report is unavailable or malformed")
    if report["communicator"] != "explicit native MPI communicator" \
            or report["implementation"] != "C++ HDF5 C API":
        raise RuntimeError(
            "native collective HDF5 capability is not the explicit-communicator C++ route")
    return _pops, report


def _rebuild_parallel_snapshot_data(
    snapshot: OutputSnapshot,
    gathered: tuple[dict[str, Any], ...],
    selected: tuple[Any, ...],
) -> dict[str, Any]:
    """Authenticate one canonical distributed snapshot from already gathered evidence."""
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
    by_token = {field.key.identity.token: field for field in selected}
    expected_tokens = set(by_token)
    malformed = [
        rank
        for rank, item in enumerate(gathered)
        if not isinstance(item["pieces"], dict)
        or set(item["pieces"]) != expected_tokens
    ]
    if malformed:
        raise ValueError(
            "collective HDF5 field ownership differs across ranks: "
            + ", ".join(map(str, malformed))
        )
    rebuilt = []
    for token in sorted(by_token):
        field = by_token[token]
        valid_boxes = tuple(snapshot.geometry(field.key).boxes)
        expected_cells = sum(
            (jhi - jlo) * (ihi - ilo)
            for jlo, ilo, jhi, ihi in valid_boxes
        )
        row = next(item for item in data["fields"] if item["key"] == field.key.to_data())
        pieces = [piece for rank in gathered for piece in rank["pieces"][token]]
        pieces.sort(key=lambda piece: (
            piece["global_box_index"], piece["owner_rank"],
            piece["array"]["content_sha256"]))
        active = []
        covered_cells = 0
        indices = set()
        for owner, item in enumerate(gathered):
            for piece in item["pieces"][token]:
                if piece["owner_rank"] != owner:
                    raise ValueError(
                        "parallel field piece owner_rank differs from its contributing rank")
                if piece["replicated"] and owner != 0:
                    raise ValueError(
                        "collective replicated field piece must use rank zero as authority")
        for piece in sorted(pieces, key=lambda value: (value["lower"], value["upper"])):
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
            box_index = piece["global_box_index"]
            if isinstance(box_index, bool) or type(box_index) is not int \
                    or box_index < 0 or box_index >= len(valid_boxes):
                raise ValueError("parallel field global_box_index is invalid")
            if box_index in indices:
                raise ValueError("parallel field global_box_index is duplicated across ranks")
            indices.add(box_index)
            if (jlo, ilo, jhi, ihi) != valid_boxes[box_index]:
                raise ValueError(
                    "parallel field piece differs from its indexed exact geometry box")
            active = [other for other in active if other[1] > jlo]
            if any(not (ihi <= other[2] or other[3] <= ilo) for other in active):
                raise ValueError("parallel field pieces overlap across ranks")
            active.append((jlo, jhi, ilo, ihi))
            covered_cells += (jhi - jlo) * (ihi - ilo)
        if covered_cells != expected_cells:
            raise ValueError(
                "parallel field pieces do not exactly cover the valid geometry boxes")
        if indices != set(range(len(valid_boxes))):
            raise ValueError(
                "parallel field pieces do not authenticate every global geometry box")
        rebuilt.append(dict(row, pieces=pieces))
    data = dict(data)
    data["fields"] = rebuilt
    return data


def _parallel_snapshot_data(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    target: Any,
    communicator: Any,
) -> tuple[dict[str, Any], tuple[Any, ...], Path, Any, dict[str, Any]]:
    from pops.output._consumer_contracts import ParallelMode
    from pops._native_collectives import allgather_value, rank, require_communicator

    selected = ()
    target_path = None
    native = None
    capability = None
    try:
        if request.parallel_mode is not ParallelMode.COLLECTIVE:
            raise ValueError(
                "a resolved communicator is valid only for HDF5 COLLECTIVE output")
        require_communicator(communicator)
        if request.rank != rank(communicator):
            raise ValueError("collective HDF5 request rank differs from its native communicator")
        native, capability = _require_native_parallel_hdf5()
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
            "rank": request.rank,
            "snapshot": canonical,
            "pieces": local_pieces,
            "preflight": {
                "target": str(target_path.expanduser().resolve()),
                "hdf5_version": capability["hdf5_version"],
                "implementation": capability["implementation"],
                "communicator": capability["communicator"],
            },
            "error": None,
        }
    except BaseException as exc:
        # Every rank must still enter the collective so one malformed local snapshot cannot leave
        # its peers blocked in allgather. The detached error is diagnostic only; no output identity
        # is derived from it.
        envelope = {
            "rank": request.rank,
            "snapshot": None,
            "pieces": None,
            "preflight": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    gathered = allgather_value(communicator, envelope)
    required_keys = {"rank", "snapshot", "pieces", "preflight", "error"}
    if len(gathered) != request.size or any(
        not isinstance(item, dict)
        or set(item) != required_keys
        or item["rank"] != rank
        for rank, item in enumerate(gathered)
    ):
        raise ValueError("collective HDF5 snapshot rank authority is malformed")
    failures = [
        "rank %d: %s" % (rank, item["error"])
        for rank, item in enumerate(gathered)
        if item["error"] is not None
    ]
    if failures:
        raise ValueError(
            "collective HDF5 snapshot preparation failed across ranks: " + "; ".join(failures)
        )
    data = None
    validation_error = None
    try:
        data = _rebuild_parallel_snapshot_data(snapshot, tuple(gathered), selected)
        if target_path is None or native is None or capability is None:
            raise RuntimeError("collective HDF5 preflight completed without local authorities")
        if not isinstance(data, dict):
            raise RuntimeError("collective HDF5 snapshot authority is not canonical data")
    except BaseException as exc:
        validation_error = "%s: %s" % (type(exc).__name__, exc)
    validation_failure = _collective_phase_error(
        communicator,
        rank=request.rank,
        size=request.size,
        phase="snapshot validation",
        error=validation_error,
    )
    if validation_failure is not None:
        raise ValueError(validation_failure)
    # The local non-None checks are inside the reported validation phase above.  No rank performs a
    # second private branch after consensus.
    return (
        cast(dict[str, Any], data),
        selected,
        cast(Path, target_path),
        native,
        cast(dict[str, Any], capability),
    )


def _parallel_temporary_path(target: Path, communicator: Any) -> _StagingAuthority:
    """Create one shared temporary on rank zero and broadcast failures without deadlocking."""
    from pops._native_collectives import allgather_value, broadcast_value, rank, size

    local_rank = rank(communicator)
    local_authority = None
    envelope: dict[str, Any] = {"path": None, "owner": None, "error": None}
    if local_rank == 0:
        try:
            local_authority = temporary_path(target)
            envelope["path"] = str(local_authority.path)
            envelope["owner"] = local_authority.owner
        except BaseException as exc:
            envelope["error"] = "%s: %s" % (type(exc).__name__, exc)

    def fail_with_cleanup(message: str) -> None:
        cleanup_error = None
        if local_rank == 0 and local_authority is not None:
            try:
                _cleanup_staging_authority(
                    local_authority,
                    replaced_message=(
                        "collective HDF5 refuses to delete a replaced temporary at %s"
                        % local_authority.path),
                )
            except BaseException as cleanup:
                cleanup_error = "%s: %s" % (type(cleanup).__name__, cleanup)
        cleanup_error = broadcast_value(communicator, cleanup_error, root=0)
        if cleanup_error is not None:
            message += "; cleanup: " + cleanup_error
        raise RuntimeError(message)

    try:
        gathered = allgather_value(communicator, envelope)
    except BaseException as error:
        cleanup_error = None
        if local_rank == 0 and local_authority is not None:
            try:
                _cleanup_staging_authority(
                    local_authority,
                    replaced_message=(
                        "collective HDF5 refuses to delete a replaced temporary at %s"
                        % local_authority.path),
                )
            except BaseException as cleanup:
                cleanup_error = "%s: %s" % (type(cleanup).__name__, cleanup)
        message = "collective HDF5 temporary-file consensus failed: %s: %s" % (
            type(error).__name__, error)
        if cleanup_error is not None:
            message += "; cleanup: " + cleanup_error
        raise RuntimeError(message) from None
    malformed = len(gathered) != size(communicator) or any(
        type(item) is not dict or set(item) != {"path", "owner", "error"}
        for item in gathered
    )
    if malformed:
        fail_with_cleanup("collective HDF5 temporary-file authority is malformed")
    else:
        failures = [
            "rank %d: %s" % (owner, item["error"])
            for owner, item in enumerate(gathered)
            if item["error"] is not None
        ]
        message = (
            "collective HDF5 temporary-file preparation failed: " + "; ".join(failures)
            if failures else None
        )
    if message is not None:
        fail_with_cleanup(message)
    authority_path = gathered[0]["path"]
    authority_owner = gathered[0]["owner"]
    if not authority_path or type(authority_owner) is not tuple \
            or len(authority_owner) != 2 \
            or any(type(item) is not int or item < 0 for item in authority_owner) \
            or any(
                item["path"] is not None or item["owner"] is not None
                for item in gathered[1:]
            ):
        fail_with_cleanup("collective HDF5 temporary-file authority is malformed")
    if local_rank == 0:
        # Rank zero authored the gathered path/owner directly from this object.  Any creation
        # failure was already in its envelope; no rank-private validation remains after consensus.
        return cast(_StagingAuthority, local_authority)
    return _StagingAuthority.observed(authority_path, authority_owner)


def _collective_temporary_owner(
    communicator: Any,
    authority: Any,
) -> tuple[int, int]:
    """Authenticate every local staging view, then fail uniformly on any owner mismatch."""
    from pops._native_collectives import allgather_value, rank, size

    local_rank = rank(communicator)
    owner = None
    error = None
    try:
        if type(authority) is _StagingAuthority:
            authority.authenticate_path()
            current = (
                os.fstat(authority.fileno())
                if local_rank == 0 else authority.path.lstat()
            )
            owner = (int(current.st_dev), int(current.st_ino))
            if owner != authority.owner:
                raise RuntimeError(
                    "local staging inode differs from its broadcast authority")
        else:
            # Retained for the MPI negative-path contract.  Every rank probes so a local
            # filesystem mismatch becomes consensus evidence instead of a one-rank branch.
            current = Path(authority).lstat()
            owner = (int(current.st_dev), int(current.st_ino))
    except BaseException as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    rows = allgather_value(communicator, {
        "rank": local_rank,
        "owner": owner,
        "error": error,
    })
    if len(rows) != size(communicator) or any(
            type(row) is not dict
            or set(row) != {"rank", "owner", "error"}
            or row["rank"] != expected_rank
            for expected_rank, row in enumerate(rows)):
        raise RuntimeError("collective HDF5 temporary inode authority is malformed")
    failures = [
        "rank %d: %s" % (expected_rank, row["error"])
        for expected_rank, row in enumerate(rows) if row["error"] is not None
    ]
    if failures:
        raise RuntimeError(
            "collective HDF5 temporary inode authentication failed: "
            + "; ".join(failures))
    root_owner = rows[0]["owner"]
    if type(root_owner) is not tuple or len(root_owner) != 2 \
            or any(type(item) is not int or item < 0 for item in root_owner) \
            or any(row["owner"] != root_owner for row in rows[1:]):
        raise RuntimeError(
            "collective HDF5 temporary inode authority differs across ranks")
    return cast(tuple[int, int], root_owner)


def _collective_phase_error(
    communicator: Any,
    *,
    rank: int,
    size: int,
    phase: str,
    error: str | None,
) -> str | None:
    """Return one identical failure string only after every rank reports the local phase."""
    from pops._native_collectives import allgather_value

    rows = allgather_value(communicator, {
        "rank": rank,
        "error": error,
    })
    if len(rows) != size or any(
            type(row) is not dict
            or set(row) != {"rank", "error"}
            or row["rank"] != expected_rank
            or (row["error"] is not None and (
                not isinstance(row["error"], str) or not row["error"]))
            for expected_rank, row in enumerate(rows)):
        return "collective HDF5 %s rank authority is malformed" % phase
    failures = [
        "rank %d: %s" % (expected_rank, row["error"])
        for expected_rank, row in enumerate(rows) if row["error"] is not None
    ]
    if failures:
        return "collective HDF5 %s failed: %s" % (phase, "; ".join(failures))
    return None


def _collective_remove(
    communicator: Any,
    authority: _StagingAuthority,
) -> str | None:
    """Remove one shared path on rank zero, then release every rank together."""
    from pops._native_collectives import barrier, rank, size

    world_rank = rank(communicator)
    local_error = None
    if world_rank == 0:
        try:
            _cleanup_staging_authority(
                authority,
                replaced_message=(
                    "collective HDF5 refuses to delete a replaced temporary at %s"
                    % authority.path),
            )
        except BaseException as exc:
            local_error = "%s: %s" % (type(exc).__name__, exc)
    else:
        try:
            authority.close()
        except BaseException as exc:
            local_error = "%s: %s" % (type(exc).__name__, exc)
    error = _collective_phase_error(
        communicator,
        rank=world_rank,
        size=size(communicator),
        phase="temporary cleanup/release",
        error=local_error,
    )
    barrier(communicator)
    return error


def _construct_collective_staged_output(
    communicator: Any,
    *,
    rank: int,
    size: int,
    authority: _StagingAuthority,
    target: Path,
    format: str,
    output_identity: Any,
    selection_identity: Any,
    verify: Any,
) -> _StagedOutputFile:
    """Construct/authenticate on every rank, then expose either success or failure uniformly."""
    staged = None
    error = None
    try:
        staged = _StagedOutputFile(
            authority,
            target,
            format=format,
            output_identity=output_identity,
            selection_identity=selection_identity,
            verify=verify,
            communicator=communicator,
        )
    except BaseException as caught:
        error = "%s: %s" % (type(caught).__name__, caught)
    failure = _collective_phase_error(
        communicator,
        rank=rank,
        size=size,
        phase="staged-file construction",
        error=error,
    )
    if failure is not None:
        cleanup_error = _collective_remove(communicator, authority)
        if cleanup_error is not None:
            failure += "; cleanup: " + cleanup_error
        raise RuntimeError(failure)
    # ``staged is None`` was itself part of the local construction try block.  A private assertion
    # here would reintroduce the rank-divergent branch this helper exists to remove.
    return cast(_StagedOutputFile, staged)


def _writer_plan(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    fields: tuple[Any, ...],
    snapshot_data: dict[str, Any],
    *,
    collective: bool,
    per_rank: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    """Build every Python-side HDF5 authority before a rank can enter HDF5."""
    arrays: dict[str, Any] = {}
    datasets: dict[str, Any] = {"fields": {}, "geometries": {}}
    evidence: dict[str, Any] = {}
    for index, field in enumerate(fields):
        name = "fields/%04d/values" % index
        datasets["fields"][field.key.identity.token] = name
        if not collective:
            from pops.output._consumer_contracts import ParallelMode

            validate_field_pieces(
                field,
                snapshot.geometry(field.key),
                complete=not per_rank,
                rank=(
                    None if request.parallel_mode is ParallelMode.ROOT else request.rank
                ),
                size=request.size,
            )
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
    for index, _field in enumerate(fields):
        name = "fields/%04d/values" % index
        global_row = snapshot_data["fields"][index]
        prefix = ([len(global_row["component_names"])]
                  if global_row["component_names"] else [])
        evidence[name] = {
            "shape": prefix + list(global_row["global_shape"]),
            "dtype": global_row["dtype"],
            "fill": "zero-outside-pieces",
            "pieces": global_row["pieces"],
        }
    evidence.update({name: array_evidence(value) for name, value in arrays.items()})
    output_manifest, identity = manifest(
        "hdf5",
        snapshot,
        request,
        evidence,
        snapshot_data=snapshot_data,
        datasets=datasets,
    )
    return arrays, datasets, output_manifest, identity


def _native_write_descriptors(
    arrays: dict[str, Any],
    fields: tuple[Any, ...],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Project exact local arrays to the typed pybind descriptors without densifying fields."""
    import numpy as np

    supported = {
        "b": {1}, "i": {1, 2, 4, 8}, "u": {1, 2, 4, 8},
        "f": {4, 8}, "c": {8, 16},
    }

    def require_array(value: Any, where: str) -> Any:
        if type(value) is not np.ndarray:
            raise TypeError("%s must be an exact NumPy array" % where)
        if not value.flags.c_contiguous or value.ndim < 1 or any(item < 1 for item in value.shape):
            raise ValueError("%s must be non-empty and C-contiguous" % where)
        if value.dtype.kind not in supported or value.dtype.itemsize not in supported[value.dtype.kind]:
            raise TypeError("%s has no supported native HDF5 scalar dtype" % where)
        return value

    root_arrays = {
        name: require_array(value, "native HDF5 root array %s" % name)
        for name, value in sorted(arrays.items())
    }
    rows = []
    for index, field in enumerate(fields):
        rows.append({
            "dataset": "fields/%04d/values" % index,
            "dtype": field.array_dtype,
            "shape": (
                ((len(field.component_names),) if field.component_names else ())
                + field.global_shape
            ),
            "pieces": tuple({
                "lower": piece.lower,
                "upper": piece.upper,
                "values": require_array(
                    piece.values, "native HDF5 field %d piece" % index),
            } for piece in field.pieces),
        })
    return root_arrays, tuple(rows)


class HDF5Writer:
    format = "hdf5"
    extension = ".h5"

    def __init__(self, mode: Any = None) -> None:
        from pops.output._consumer_contracts import ParallelMode

        if mode is None:
            mode = ParallelMode.SERIAL
        if type(mode) is not ParallelMode:
            raise TypeError("HDF5Writer mode must be an exact ParallelMode")
        self._mode = mode

    def preflight(self, execution_context: Any) -> dict[str, Any]:
        """Return the installed HDF5 capability before native engine construction."""
        from pops.output._consumer_contracts import ParallelMode

        if type(self._mode) is not ParallelMode:
            raise RuntimeError("HDF5Writer preflight requires its resolved format mode")
        result = writer_execution_capability(
            execution_context,
            self._mode,
            provider_id="pops.output.hdf5.v1",
        )
        if self._mode is ParallelMode.COLLECTIVE:
            _native, capability = _require_native_parallel_hdf5()
            result["library"] = dict(capability)
            result["collective_dataset_transfer"] = True
        else:
            h5py = _require_h5py(False)
            result["library"] = {
                "h5py": str(getattr(h5py, "__version__", "unknown")),
                "hdf5": str(getattr(getattr(h5py, "version", None),
                                    "hdf5_version", "unknown")),
                "mpi": False,
            }
        return result

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
            raise ValueError("HDF5 writer mode differs from its resolved output request")
        if request.parallel_mode is ParallelMode.SERIAL and communicator is not None:
            raise ValueError("SERIAL HDF5 writer session cannot carry a communicator")
        detached_root = request.parallel_mode is ParallelMode.ROOT and request.rank == 0
        if request.parallel_mode is not ParallelMode.SERIAL and communicator is None \
                and not detached_root:
            raise TypeError(
                "distributed HDF5 writer session requires its communicator unless a complete "
                "ROOT snapshot was detached for post-commit writing")
        authority = writer_session_authority(self.format, request, target)

        def stage_file() -> _StagedOutputFile:
            return self._stage_file(
                snapshot,
                request,
                target,
                communicator=(
                    communicator
                    if request.parallel_mode is ParallelMode.COLLECTIVE else None
                ),
            )

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
        *,
        communicator: Any = None,
    ) -> _StagedOutputFile:
        from pops.output._consumer_contracts import ParallelMode

        collective = request.parallel_mode is ParallelMode.COLLECTIVE
        per_rank = request.parallel_mode is ParallelMode.PER_RANK
        world_rank = None
        if communicator is not None:
            from pops._native_collectives import rank as native_rank

            if not collective:
                raise ValueError(
                    "HDF5 communicator is valid only for an exact COLLECTIVE request")
            world_rank = native_rank(communicator)
            snapshot_data, fields, target, native, _native_capability = _parallel_snapshot_data(
                snapshot, request, target, communicator
            )
        elif collective:
            raise TypeError("collective HDF5 requires the resolved communicator")
        else:
            if request.parallel_mode is ParallelMode.ROOT and request.rank != 0:
                raise ValueError("ROOT HDF5 writer may be invoked only on rank 0")
            fields = snapshot.select(request)
            snapshot_data = snapshot.to_data(request)
            native = None
            h5py = _require_h5py(False)
            target = Path(target)
            if target.suffix not in {".h5", ".hdf5"}:
                raise ValueError("HDF5 target must end in .h5 or .hdf5")
        plan = None
        plan_manifest = None
        plan_identity = None
        plan_error = None
        try:
            plan = _writer_plan(
                snapshot,
                request,
                fields,
                snapshot_data,
                collective=collective,
                per_rank=per_rank,
            )
            if type(plan) is not tuple or len(plan) != 4:
                raise RuntimeError("HDF5 writer plan has an invalid result schema")
            plan_manifest = plan[2]
            plan_identity = plan[3].token
        except BaseException as exc:
            plan_error = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            from pops._native_collectives import allgather_value

            plan_rows = allgather_value(communicator, {
                "rank": world_rank,
                "manifest": plan_manifest,
                "identity": plan_identity,
                "error": plan_error,
            })
            if len(plan_rows) != request.size or any(
                not isinstance(row, dict)
                or set(row) != {"rank", "manifest", "identity", "error"}
                or row["rank"] != owner
                for owner, row in enumerate(plan_rows)
            ):
                raise RuntimeError(
                    "collective HDF5 writer-plan authority is malformed")
            failures = [
                "rank %d: %s" % (owner, row["error"])
                for owner, row in enumerate(plan_rows) if row["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    "collective HDF5 writer-plan preparation failed: " + "; ".join(failures))
            authority = (plan_rows[0]["manifest"], plan_rows[0]["identity"])
            if any(
                (row["manifest"], row["identity"]) != authority
                for row in plan_rows[1:]
            ):
                raise RuntimeError(
                    "collective HDF5 manifest/identity differs across ranks")
        elif plan_error is not None:
            raise RuntimeError("HDF5 writer-plan preparation failed: " + plan_error)
        # Invalid/None local plans were reported in ``plan_error`` before rank consensus.
        arrays, _datasets, output_manifest, identity = cast(tuple[Any, ...], plan)
        temporary = (
            _parallel_temporary_path(target, communicator)
            if collective
            else temporary_path(target)
        )
        if collective:
            try:
                _collective_temporary_owner(communicator, temporary)
            except BaseException as error:
                cleanup_error = _collective_remove(communicator, temporary)
                message = "collective HDF5 temporary inode authentication failed: %s: %s" % (
                    type(error).__name__, error)
                if cleanup_error is not None:
                    message += "; cleanup: " + cleanup_error
                raise RuntimeError(message) from None
            from pops._native_collectives import allgather_value

            descriptors = None
            root_arrays = None
            field_rows = None
            descriptor_error = None
            try:
                descriptors = _native_write_descriptors(arrays, fields)
                if type(descriptors) is not tuple or len(descriptors) != 2:
                    raise RuntimeError(
                        "collective HDF5 native descriptors have an invalid result schema"
                    )
                root_arrays, field_rows = descriptors
                if root_arrays is None or field_rows is None or native is None:
                    raise RuntimeError(
                        "collective HDF5 descriptor preparation returned no native plan")
            except BaseException as exc:
                descriptor_error = "%s: %s" % (type(exc).__name__, exc)
            descriptor_failure = _collective_phase_error(
                communicator,
                rank=cast(int, world_rank),
                size=request.size,
                phase="descriptor preparation",
                error=descriptor_error,
            )
            if descriptor_failure is not None:
                cleanup_error = _collective_remove(communicator, temporary)
                message = descriptor_failure
                if cleanup_error is not None:
                    message += "; cleanup: " + cleanup_error
                raise RuntimeError(message)
            # Every local non-None check was part of ``descriptor_error`` above.
            native_writer = cast(Any, native)
            native_error = None
            try:
                native_writer._write_parallel_hdf5(
                    communicator,
                    str(temporary.path),
                    json_text(output_manifest),
                    root_arrays,
                    field_rows,
                )
            except BaseException as exc:
                native_error = "%s: %s" % (type(exc).__name__, exc)
            native_failure = _collective_phase_error(
                communicator,
                rank=cast(int, world_rank),
                size=request.size,
                phase="native write",
                error=native_error,
            )
            if native_failure is not None:
                cleanup_error = _collective_remove(communicator, temporary)
                message = native_failure
                if cleanup_error is not None:
                    message += "; cleanup: " + cleanup_error
                raise RuntimeError(message)
        else:
            io_error = None
            try:
                with os.fdopen(temporary.duplicate(), "r+b") as staging_file:
                    with h5py.File(staging_file, "w") as output:
                        output.attrs["pops_output_manifest"] = json_text(output_manifest)
                        for name, value in arrays.items():
                            output.create_dataset(name, data=value, compression="gzip")
                        for index, field in enumerate(fields):
                            name = "fields/%04d/values" % index
                            shape = (
                                ((len(field.component_names),) if field.component_names else ())
                                + field.global_shape
                            )
                            dataset = output.require_dataset(
                                name, shape=shape, dtype=field.array_dtype)
                            for piece in field.pieces:
                                jlo, ilo = piece.lower
                                jhi, ihi = piece.upper
                                dataset[..., jlo:jhi, ilo:ihi] = piece.values
                        output.flush()
            except BaseException as exc:
                io_error = "%s: %s" % (type(exc).__name__, exc)
            if io_error is not None:
                cleanup_error = None
                try:
                    _cleanup_staging_authority(
                        temporary,
                        replaced_message=(
                            "HDF5 staging cleanup refused a replaced temporary at %s"
                            % temporary.path),
                    )
                except BaseException as cleanup:
                    cleanup_error = "%s: %s" % (type(cleanup).__name__, cleanup)
                if cleanup_error is not None:
                    io_error += "; cleanup: " + cleanup_error
                raise RuntimeError("HDF5 write failed: %s" % io_error)
        failure = None
        if communicator is None or world_rank == 0:
            try:
                read_hdf5(temporary.path).require_selection(request)
            except BaseException as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            failure = _collective_phase_error(
                communicator,
                rank=cast(int, world_rank),
                size=request.size,
                phase="native verification",
                error=failure,
            )
        if failure is not None:
            if communicator is not None:
                cleanup_error = _collective_remove(communicator, temporary)
                if cleanup_error is not None:
                    failure += "; cleanup: " + cleanup_error
            else:
                try:
                    _cleanup_staging_authority(
                        temporary,
                        replaced_message=(
                            "HDF5 verification cleanup refused a replaced temporary at %s"
                            % temporary.path),
                    )
                except BaseException as cleanup:
                    failure += "; cleanup: %s: %s" % (
                        type(cleanup).__name__, cleanup)
            raise RuntimeError("prepared HDF5 failed native verification: %s" % failure)
        if communicator is not None:
            return _construct_collective_staged_output(
                communicator,
                rank=cast(int, world_rank),
                size=request.size,
                authority=temporary,
                target=target,
                format=self.format,
                output_identity=identity,
                selection_identity=request.publication_identity,
                verify=read_hdf5,
            )
        return _StagedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.publication_identity,
            verify=read_hdf5,
            communicator=communicator,
        )


def read_hdf5(path: Any) -> ReopenedOutput:
    import numpy as np

    h5py = _require_h5py(False)
    with h5py.File(path, "r") as source:
        if set(source.attrs) != {"pops_output_manifest"}:
            raise ValueError("HDF5 has no PoPS scientific output manifest")
        output_manifest, identity = authenticate_manifest(
            json.loads(source.attrs["pops_output_manifest"]), "hdf5")
        declared_datasets = set(output_manifest["arrays"])
        declared_groups = {
            prefix
            for name in declared_datasets
            for prefix in (
                "/".join(name.split("/")[:depth])
                for depth in range(1, len(name.split("/")))
            )
        }
        actual_datasets: set[str] = set()
        actual_groups: set[str] = set()
        attribute_errors: list[str] = []

        def inventory(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                actual_datasets.add(name)
            elif isinstance(value, h5py.Group):
                actual_groups.add(name)
            else:
                attribute_errors.append("unsupported object %s" % name)
            if len(value.attrs) != 0:
                attribute_errors.append("unexpected attributes on %s" % name)

        source.visititems(inventory)
        if attribute_errors:
            raise ValueError("HDF5 object metadata differs from its exact manifest: "
                             + "; ".join(attribute_errors))
        if actual_datasets != declared_datasets or actual_groups != declared_groups:
            raise ValueError("HDF5 datasets/groups differ from its exact manifest")
        arrays = {}
        for name, evidence in output_manifest["arrays"].items():
            value = np.asarray(source[name][...])
            arrays[name] = value
            if "pieces" in evidence:
                if set(evidence) != {"shape", "dtype", "fill", "pieces"} \
                        or evidence["fill"] != "zero-outside-pieces":
                    raise ValueError("HDF5 field evidence schema is not exact")
                if list(value.shape) != evidence["shape"] \
                        or value.dtype.str != evidence["dtype"]:
                    raise ValueError("HDF5 field shape/dtype differs from its manifest")
                spatial_shape = tuple(evidence["shape"][-2:])
                written = np.zeros(spatial_shape, dtype=np.bool_)
                for piece in evidence["pieces"]:
                    jlo, ilo = piece["lower"]
                    jhi, ihi = piece["upper"]
                    if np.any(written[jlo:jhi, ilo:ihi]):
                        raise ValueError("HDF5 manifest field pieces overlap")
                    written[jlo:jhi, ilo:ihi] = True
                    if array_evidence(value[..., jlo:jhi, ilo:ihi]) != piece["array"]:
                        raise ValueError("HDF5 parallel piece failed verification")
                gap = np.ascontiguousarray(value[..., ~written])
                if gap.tobytes() != bytes(gap.nbytes):
                    raise ValueError("HDF5 field has noncanonical data outside declared pieces")
            elif array_evidence(value) != evidence:
                raise ValueError("HDF5 dataset %r failed verification" % name)
    return ReopenedOutput(output_manifest, arrays, identity)


__all__ = ["HDF5Writer", "read_hdf5"]
