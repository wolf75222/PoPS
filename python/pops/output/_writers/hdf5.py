"""Exact SERIAL, ROOT, COLLECTIVE, and PER_RANK HDF5 scientific-output backend."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from pops.output._writers.common import (
    OutputWriterSession,
    ReopenedOutput,
    _StagedOutputFile,
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
    if report["communicator"] != "MPI_COMM_WORLD" \
            or report["implementation"] != "C++ HDF5 C API":
        raise RuntimeError("native collective HDF5 capability is not the final C++ world route")
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
    from pops._native_collectives import allgather_value, rank, require_world

    selected = ()
    target_path = None
    native = None
    capability = None
    try:
        if request.parallel_mode is not ParallelMode.COLLECTIVE:
            raise ValueError(
                "a resolved communicator is valid only for HDF5 COLLECTIVE output")
        require_world(communicator)
        if request.rank != rank(communicator):
            raise ValueError("collective HDF5 request rank differs from native MPI_COMM_WORLD")
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
    except Exception as exc:
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
    except Exception as exc:
        validation_error = "%s: %s" % (type(exc).__name__, exc)
    validation_errors = allgather_value(communicator, validation_error)
    validation_failures = [
        "rank %d: %s" % (rank, error)
        for rank, error in enumerate(validation_errors) if error is not None
    ]
    if validation_failures:
        raise ValueError(
            "collective HDF5 snapshot validation failed across ranks: "
            + "; ".join(validation_failures)
        )
    if data is None or target_path is None or native is None or capability is None:
        raise RuntimeError("collective HDF5 snapshot validation returned no authority")
    return data, selected, target_path, native, capability


def _parallel_temporary_path(target: Path, communicator: Any) -> Path:
    """Create one shared temporary on rank zero and broadcast failures without deadlocking."""
    from pops._native_collectives import allgather_value, rank

    local_rank = rank(communicator)
    envelope: dict[str, str | None] = {"path": None, "error": None}
    if local_rank == 0:
        try:
            envelope["path"] = str(temporary_path(target))
        except Exception as exc:
            envelope["error"] = "%s: %s" % (type(exc).__name__, exc)
    gathered = allgather_value(communicator, envelope)
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


def _collective_temporary_owner(
    communicator: Any,
    path: Path,
) -> tuple[int, int]:
    """Authenticate the shared staging inode without letting rank-zero I/O skip consensus."""
    from pops._native_collectives import broadcast_value, rank

    envelope: dict[str, Any] | None = None
    if rank(communicator) == 0:
        try:
            stat = path.lstat()
            envelope = {
                "owner": (int(stat.st_dev), int(stat.st_ino)),
                "error": None,
            }
        except BaseException as exc:
            envelope = {
                "owner": None,
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
    envelope = broadcast_value(communicator, envelope, root=0)
    if type(envelope) is not dict or set(envelope) != {"owner", "error"}:
        raise RuntimeError(
            "collective HDF5 temporary inode authority is malformed"
        )
    error = envelope["error"]
    if error is not None:
        if not isinstance(error, str) or not error:
            raise RuntimeError(
                "collective HDF5 temporary inode failure is malformed"
            )
        raise RuntimeError(
            "collective HDF5 temporary inode authentication failed: rank 0: " + error
        )
    owner = envelope["owner"]
    if type(owner) is not tuple or len(owner) != 2 \
            or any(type(item) is not int or item < 0 for item in owner):
        raise RuntimeError(
            "collective HDF5 temporary inode authority is malformed"
        )
    return owner


def _collective_remove(
    communicator: Any,
    path: Path,
    expected_owner: tuple[int, int],
) -> str | None:
    """Remove one shared path on rank zero, then release every rank together."""
    from pops._native_collectives import barrier, broadcast_value, rank

    error = None
    if rank(communicator) == 0:
        try:
            try:
                current = path.lstat()
            except FileNotFoundError:
                current = None
            if current is not None:
                owner = (int(current.st_dev), int(current.st_ino))
                if owner != expected_owner:
                    raise RuntimeError(
                        "collective HDF5 refuses to delete a replaced temporary at %s" % path)
                path.unlink()
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
    error = broadcast_value(communicator, error, root=0)
    barrier(communicator)
    return error


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
        if request.parallel_mode is not ParallelMode.SERIAL and communicator is None:
            raise TypeError("distributed HDF5 writer session requires its communicator")
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
        except Exception as exc:
            plan_error = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            from pops._native_collectives import allgather_value

            plan_rows = allgather_value(communicator, {
                "rank": world_rank,
                "manifest": plan_manifest,
                "identity": plan_identity,
                "error": plan_error,
            })
            if any(
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
        if plan is None:
            raise RuntimeError("HDF5 writer-plan preparation returned no exact authority")
        arrays, _datasets, output_manifest, identity = plan
        temporary = (
            _parallel_temporary_path(target, communicator)
            if collective
            else temporary_path(target)
        )
        temporary_owner = None
        if collective:
            temporary_owner = _collective_temporary_owner(communicator, temporary)
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
            except Exception as exc:
                descriptor_error = "%s: %s" % (type(exc).__name__, exc)
            consensus = allgather_value(communicator, descriptor_error)
            failures = [
                "rank %d: %s" % (owner, error)
                for owner, error in enumerate(consensus) if error is not None
            ]
            if failures:
                cleanup_error = _collective_remove(
                    communicator, temporary, temporary_owner)
                message = "collective HDF5 descriptor preparation failed: " + "; ".join(failures)
                if cleanup_error is not None:
                    message += "; cleanup: " + cleanup_error
                raise RuntimeError(message)
            if native is None or root_arrays is None or field_rows is None:
                raise RuntimeError(
                    "collective HDF5 descriptor preparation returned no native plan")
            try:
                native._write_parallel_hdf5(
                    communicator,
                    str(temporary),
                    json_text(output_manifest),
                    root_arrays,
                    field_rows,
                )
            except Exception as exc:
                cleanup_error = _collective_remove(
                    communicator, temporary, temporary_owner)
                message = "collective HDF5 native transaction failed: %s: %s" % (
                    type(exc).__name__, exc)
                if cleanup_error is not None:
                    message += "; cleanup: " + cleanup_error
                raise RuntimeError(message) from None
        else:
            io_error = None
            try:
                with h5py.File(temporary, "w") as output:
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
            except Exception as exc:
                io_error = "%s: %s" % (type(exc).__name__, exc)
            if io_error is not None:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("HDF5 write failed: %s" % io_error)
        if communicator is not None:
            from pops._native_collectives import barrier

            barrier(communicator)
        failure = None
        if communicator is None or world_rank == 0:
            try:
                read_hdf5(temporary).require_selection(request)
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
        if communicator is not None:
            from pops._native_collectives import broadcast_value

            failure = broadcast_value(communicator, failure, root=0)
        if failure is not None:
            if communicator is not None:
                if temporary_owner is None:
                    raise RuntimeError(
                        "collective HDF5 verification cleanup has no temporary inode authority")
                cleanup_error = _collective_remove(
                    communicator, temporary, temporary_owner)
                if cleanup_error is not None:
                    failure += "; cleanup: " + cleanup_error
            else:
                temporary.unlink(missing_ok=True)
            raise RuntimeError("prepared HDF5 failed native verification: %s" % failure)
        if communicator is not None:
            barrier(communicator)
        return _StagedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.publication_identity,
            verify=read_hdf5,
            communicator=communicator,
            temporary_owner=temporary_owner,
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
