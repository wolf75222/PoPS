"""Exact format consumers with prepare/verify/publish transactions."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import struct
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pops.identity import Identity, make_identity

from .data import OutputRequest, OutputSnapshot, array_evidence


OUTPUT_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _identity_from_token(token: Any, domain: str, where: str) -> Identity:
    try:
        result = Identity.from_token(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s has an invalid identity" % where) from exc
    if result.domain != domain:
        raise ValueError("%s must use the %r identity domain" % (where, domain))
    return result


def deterministic_target(directory: Any, prefix: Any, request: OutputRequest,
                         snapshot: OutputSnapshot, extension: str) -> Path:
    """Return the sole deterministic scientific-output filename."""
    root = Path(directory)
    clean_prefix = _SAFE_NAME.sub("-", str(prefix)).strip("-")
    clean_consumer = _SAFE_NAME.sub("-", request.consumer_id).strip("-")
    clean_clock = _SAFE_NAME.sub("-", snapshot.clock.clock_id).strip("-")
    if not clean_prefix or not clean_consumer or not clean_clock:
        raise ValueError("output filename parts must contain a safe non-empty token")
    if not extension.startswith(".") or "/" in extension or "\\" in extension:
        raise ValueError("output extension must be a simple suffix")
    name = "%s__%s__%s__s%09d__%s%s" % (
        clean_prefix, clean_consumer, clean_clock, snapshot.clock.macro_step,
        request.identity.hexdigest[:16], extension)
    return root / name


def _manifest(format_name: str, snapshot: OutputSnapshot, request: OutputRequest,
              arrays: dict[str, Any], *, snapshot_data: dict[str, Any] | None = None,
              datasets: dict[str, Any] | None = None) -> tuple[dict[str, Any], Identity]:
    base = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "format": format_name,
        "snapshot": snapshot_data if snapshot_data is not None else snapshot.to_data(request),
        "datasets": datasets or {},
        "arrays": {name: arrays[name] for name in sorted(arrays)},
    }
    identity = make_identity("scientific-output", base)
    return dict(base, output_identity=identity.token), identity


def _authenticate_manifest(value: Any, format_name: str) -> tuple[dict[str, Any], Identity]:
    if not isinstance(value, dict):
        raise TypeError("scientific output manifest must be a mapping")
    required = {"schema_version", "format", "snapshot", "datasets", "arrays", "output_identity"}
    if set(value) != required:
        raise ValueError("scientific output manifest keys are not exact")
    if value["schema_version"] != OUTPUT_SCHEMA_VERSION or value["format"] != format_name:
        raise ValueError("scientific output schema/format mismatch")
    supplied = _identity_from_token(value["output_identity"], "scientific-output",
                                    "output_identity")
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

    __slots__ = ("temporary", "target", "format", "output_identity", "selection_identity",
                 "_verify", "_published", "_discarded", "_created_target", "_communicator")

    def __init__(self, temporary: Any, target: Any, *, format: str,
                 output_identity: Identity, selection_identity: Identity,
                 verify: Callable[[Any], Any], communicator: Any = None) -> None:
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
            except Exception as exc:  # synchronized below before any rank leaves publication
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


def _temporary(target: Path, communicator: Any = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    rank = 0 if communicator is None else int(communicator.Get_rank())
    path = None
    if rank == 0:
        fd, name = tempfile.mkstemp(
            prefix=".%s." % target.name, suffix=".prepared", dir=str(target.parent))
        os.close(fd)
        path = name
    if communicator is not None:
        path = communicator.bcast(path, root=0)
    return Path(path)


def _serial_payload(snapshot: OutputSnapshot, request: OutputRequest) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]]:
    arrays, datasets = {}, {"fields": {}, "geometries": {}}
    fields = snapshot.select(request)
    for index, field in enumerate(fields):
        name = "field_%04d" % index
        arrays[name] = field.materialize()
        datasets["fields"][field.key.identity.token] = name
    geometries = _selected_geometries(snapshot, request, fields)
    for index, geometry in enumerate(sorted(geometries.values(), key=lambda item: item.key)):
        coverage = "geometry_%04d_coverage" % index
        valid = "geometry_%04d_valid" % index
        volumes = "geometry_%04d_volumes" % index
        arrays[coverage], arrays[valid], arrays[volumes] = (
            geometry.coverage, geometry.valid_cells, geometry.cell_volumes)
        datasets["geometries"]["%s#%d" % geometry.key] = {
            "coverage": coverage, "valid_cells": valid, "cell_volumes": volumes,
        }
    evidence = {name: array_evidence(value) for name, value in arrays.items()}
    return arrays, datasets, evidence


def _selected_geometries(snapshot: OutputSnapshot, request: OutputRequest,
                         fields: Any) -> dict[Any, Any]:
    geometries = {snapshot.geometry(field.key).key: snapshot.geometry(field.key) for field in fields}
    diagnostic_layouts = {item.layout_identity.token for item in request.diagnostics}
    geometries.update({item.key: item for item in snapshot.geometries
                       if item.layout_identity.token in diagnostic_layouts})
    return geometries


class NPZWriter:
    format = "npz"
    extension = ".npz"

    def prepare(self, snapshot: OutputSnapshot, request: OutputRequest,
                target: Any, *, communicator: Any = None) -> PreparedOutputFile:
        if communicator is not None:
            raise ValueError("serial NPZ output cannot carry a communicator")
        if request.parallel:
            raise ValueError("NPZ has no collective writer; select HDF5(parallel=True)")
        import numpy as np

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("NPZ target must end in .npz")
        arrays, datasets, evidence = _serial_payload(snapshot, request)
        manifest, identity = _manifest(self.format, snapshot, request, evidence, datasets=datasets)
        temporary = _temporary(target)
        payload = dict(arrays)
        payload["pops_output_manifest"] = np.asarray(_json(manifest))
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        read_npz(temporary).require_selection(request)
        return PreparedOutputFile(
            temporary, target, format=self.format, output_identity=identity,
            selection_identity=request.identity, verify=read_npz)


def read_npz(path: Any) -> ReopenedOutput:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if "pops_output_manifest" not in files:
            raise ValueError("NPZ has no PoPS scientific output manifest")
        manifest, identity = _authenticate_manifest(
            json.loads(str(data["pops_output_manifest"])), "npz")
        expected = set(manifest["arrays"]) | {"pops_output_manifest"}
        if files != expected:
            raise ValueError("NPZ keys differ from its exact output manifest")
        arrays = {name: np.asarray(data[name]).copy() for name in manifest["arrays"]}
    for name, evidence in manifest["arrays"].items():
        if array_evidence(arrays[name]) != evidence:
            raise ValueError("NPZ array %r failed content verification" % name)
    return ReopenedOutput(manifest, arrays, identity)


def _require_h5py(parallel: bool) -> Any:
    try:
        import h5py
    except ImportError:
        raise RuntimeError("HDF5 output requires the optional h5py dependency") from None
    if parallel and not h5py.get_config().mpi:
        raise RuntimeError(
            "collective HDF5 requires h5py built with MPI; parallel=False is the serial route")
    return h5py


def _parallel_snapshot_data(snapshot: OutputSnapshot, request: OutputRequest,
                            communicator: Any) -> dict[str, Any]:
    local = {field.key.identity.token: [piece.to_data() for piece in field.pieces]
             for field in snapshot.select(request)}
    gathered = communicator.allgather(local)
    data = snapshot.to_data(request)
    by_token = {field.key.identity.token: field for field in snapshot.select(request)}
    rebuilt = []
    for token in sorted(by_token):
        field = by_token[token]
        row = next(item for item in data["fields"] if item["key"] == field.key.to_data())
        pieces = [piece for rank in gathered for piece in rank[token]]
        pieces.sort(key=lambda piece: (piece["lower"], piece["upper"],
                                       piece["array"]["content_sha256"]))
        active = []
        covered_cells = 0
        for piece in pieces:
            jlo, ilo = piece["lower"]
            jhi, ihi = piece["upper"]
            if (jlo < 0 or ilo < 0 or jhi <= jlo or ihi <= ilo
                    or jhi > field.global_shape[0] or ihi > field.global_shape[1]):
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
    return data


class HDF5Writer:
    format = "hdf5"
    extension = ".h5"

    def prepare(self, snapshot: OutputSnapshot, request: OutputRequest, target: Any,
                *, communicator: Any = None) -> PreparedOutputFile:
        h5py = _require_h5py(request.parallel)
        if request.parallel:
            required = ("Get_rank", "bcast", "allgather", "Barrier")
            if communicator is None or any(not callable(getattr(communicator, name, None))
                                           for name in required):
                raise TypeError("collective HDF5 requires the resolved communicator")
        elif communicator is not None:
            raise ValueError("a communicator is valid only for HDF5 parallel output")
        target = Path(target)
        if target.suffix not in {".h5", ".hdf5"}:
            raise ValueError("HDF5 target must end in .h5 or .hdf5")
        fields = snapshot.select(request)
        snapshot_data = (_parallel_snapshot_data(snapshot, request, communicator)
                         if request.parallel else snapshot.to_data(request))
        arrays, datasets, evidence = {}, {"fields": {}, "geometries": {}}, {}
        for index, field in enumerate(fields):
            name = "fields/%04d/values" % index
            datasets["fields"][field.key.identity.token] = name
            if not request.parallel:
                arrays[name] = field.materialize()
        geometries = _selected_geometries(snapshot, request, fields)
        for index, geometry in enumerate(sorted(geometries.values(), key=lambda item: item.key)):
            coverage = "geometry/%04d/coverage" % index
            valid = "geometry/%04d/valid_cells" % index
            volumes = "geometry/%04d/cell_volumes" % index
            arrays[coverage], arrays[valid], arrays[volumes] = (
                geometry.coverage, geometry.valid_cells, geometry.cell_volumes)
            datasets["geometries"]["%s#%d" % geometry.key] = {
                "coverage": coverage, "valid_cells": valid, "cell_volumes": volumes,
            }
        if request.parallel:
            for index, _field in enumerate(fields):
                name = "fields/%04d/values" % index
                global_row = snapshot_data["fields"][index]
                evidence[name] = {"pieces": global_row["pieces"]}
        evidence.update({name: array_evidence(value) for name, value in arrays.items()})
        manifest, identity = _manifest(
            self.format, snapshot, request, evidence, snapshot_data=snapshot_data,
            datasets=datasets)
        temporary = _temporary(target, communicator)
        options = ({"driver": "mpio", "comm": communicator} if request.parallel else {})
        rank = 0 if communicator is None else int(communicator.Get_rank())
        with h5py.File(temporary, "w", **options) as output:
            output.attrs["pops_output_manifest"] = _json(manifest)
            for name, value in arrays.items():
                if request.parallel:
                    dataset = output.create_dataset(name, shape=value.shape, dtype=value.dtype)
                    if rank == 0:
                        dataset[...] = value
                else:
                    output.create_dataset(name, data=value, compression="gzip")
            for index, field in enumerate(fields):
                name = "fields/%04d/values" % index
                shape = ((len(field.component_names),) if field.component_names else ()) + field.global_shape
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
            temporary, target, format=self.format, output_identity=identity,
            selection_identity=request.identity, verify=read_hdf5, communicator=communicator)


def read_hdf5(path: Any) -> ReopenedOutput:
    import numpy as np
    h5py = _require_h5py(False)

    with h5py.File(path, "r") as source:
        if "pops_output_manifest" not in source.attrs:
            raise ValueError("HDF5 has no PoPS scientific output manifest")
        manifest, identity = _authenticate_manifest(
            json.loads(source.attrs["pops_output_manifest"]), "hdf5")
        arrays = {}
        for name, evidence in manifest["arrays"].items():
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
        declared_roots = {name.split("/", 1)[0] for name in manifest["arrays"]}
        if set(source.keys()) != declared_roots:
            raise ValueError("HDF5 datasets differ from its exact manifest")
    return ReopenedOutput(manifest, arrays, identity)


_VTK_TYPES = {"<f8": "Float64", "<f4": "Float32", "<i8": "Int64", "<i4": "Int32",
              "|u1": "UInt8"}


def _vtk_array(value: Any) -> tuple[str, str]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and struct.pack("=I", 1)[0] != 1):
        array = array.byteswap().newbyteorder("<")
    elif array.dtype.byteorder == "=":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    vtk_type = _VTK_TYPES.get(array.dtype.str)
    if vtk_type is None:
        raise TypeError("ParaView writer does not support dtype %s" % array.dtype)
    raw = memoryview(array).cast("B").tobytes()
    return vtk_type, base64.b64encode(struct.pack("<I", len(raw)) + raw).decode("ascii")


def _decode_vtk_array(node: ET.Element) -> Any:
    import numpy as np

    types = {value: np.dtype(key) for key, value in _VTK_TYPES.items()}
    dtype = types[node.attrib["type"]]
    raw = base64.b64decode((node.text or "").strip())
    if len(raw) < 4 or struct.unpack("<I", raw[:4])[0] != len(raw) - 4:
        raise ValueError("invalid VTK inline binary array")
    values = np.frombuffer(raw[4:], dtype=dtype).copy()
    components = int(node.attrib.get("NumberOfComponents", "1"))
    return values.reshape((-1, components)) if components != 1 else values


class ParaViewWriter:
    """Single-file VTK UnstructuredGrid consumer readable directly by ParaView."""
    format = "paraview-vtu"
    extension = ".vtu"

    def prepare(self, snapshot: OutputSnapshot, request: OutputRequest,
                target: Any, *, communicator: Any = None) -> PreparedOutputFile:
        if communicator is not None:
            raise ValueError("serial ParaView output cannot carry a communicator")
        if request.parallel:
            raise ValueError("ParaView VTU publication is serial; use collective HDF5 for parallel IO")
        import numpy as np

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("ParaView target must end in .vtu")
        fields = snapshot.select(request)
        if any(field.centering != "cell" for field in fields):
            raise NotImplementedError(
                "ParaView VTU currently proves cell-centered arrays only; no centering substitution")
        geometries = []
        seen = set()
        for field in fields:
            geometry = snapshot.geometry(field.key)
            if geometry.key not in seen:
                seen.add(geometry.key)
                geometries.append(geometry)
        for geometry in _selected_geometries(snapshot, request, fields).values():
            if geometry.key not in seen:
                seen.add(geometry.key)
                geometries.append(geometry)
        geometries.sort(key=lambda item: item.key)
        points, layout_ordinals, levels, covered, volumes = [], [], [], [], []
        offsets = {}
        cell = 0
        for ordinal, geometry in enumerate(geometries):
            start = cell
            ny, nx = geometry.cell_shape
            ox, oy = geometry.origin
            dx, dy = geometry.spacing
            for j in range(ny):
                for i in range(nx):
                    if not geometry.valid_cells[j, i]:
                        continue
                    x0, x1 = ox + i * dx, ox + (i + 1) * dx
                    y0, y1 = oy + j * dy, oy + (j + 1) * dy
                    points.extend(((x0, y0, 0.0), (x1, y0, 0.0),
                                   (x1, y1, 0.0), (x0, y1, 0.0)))
                    layout_ordinals.append(ordinal)
                    levels.append(geometry.level)
                    covered.append(int(geometry.coverage[j, i]))
                    volumes.append(float(geometry.cell_volumes[j, i]))
                    cell += 1
            offsets[geometry.key] = (start, cell, ordinal)
        n_cells = cell
        arrays = {
            "Points": np.asarray(points, dtype="<f8"),
            "connectivity": np.arange(n_cells * 4, dtype="<i8"),
            "offsets": np.arange(4, n_cells * 4 + 1, 4, dtype="<i8"),
            "types": np.full(n_cells, 9, dtype="u1"),
            "pops_layout": np.asarray(layout_ordinals, dtype="<i4"),
            "pops_level": np.asarray(levels, dtype="<i4"),
            "pops_coverage": np.asarray(covered, dtype="u1"),
            "pops_cell_volume": np.asarray(volumes, dtype="<f8"),
        }
        datasets = {"fields": {}, "geometries": {}}
        for ordinal, geometry in enumerate(geometries):
            datasets["geometries"]["%s#%d" % geometry.key] = {
                "layout_ordinal": ordinal, "cell_range": list(offsets[geometry.key][:2]),
            }
        for index, field in enumerate(fields):
            name = "field_%04d" % index
            dense = field.materialize()
            valid = snapshot.geometry(field.key).valid_cells.ravel()
            components = len(field.component_names) or 1
            start, end, ordinal = offsets[snapshot.geometry(field.key).key]
            if field.component_names:
                combined = np.zeros((n_cells, components), dtype=dense.dtype)
                values = dense.reshape((components, -1)).T[valid]
                combined[start:end, :] = values
            else:
                combined = np.zeros(n_cells, dtype=dense.dtype)
                values = dense.reshape(-1)[valid]
                combined[start:end] = values
            arrays[name] = combined
            datasets["fields"][field.key.identity.token] = {
                "name": name, "layout_ordinal": ordinal, "cell_range": [start, end],
                "array": array_evidence(values),
            }
        evidence = {name: array_evidence(value) for name, value in arrays.items()}
        manifest, identity = _manifest(
            self.format, snapshot, request, evidence, datasets=datasets)
        encoded_arrays = {name: _vtk_array(value) for name, value in arrays.items()}
        temporary = _temporary(target)
        point_type, point_data = encoded_arrays["Points"]
        cell_arrays = []
        for name in ("pops_layout", "pops_level", "pops_coverage", "pops_cell_volume") + tuple(
                "field_%04d" % index for index in range(len(fields))):
            vtk_type, encoded = encoded_arrays[name]
            components = arrays[name].shape[1] if arrays[name].ndim == 2 else 1
            cell_arrays.append(
                '<DataArray type="%s" Name="%s" NumberOfComponents="%d" format="binary">%s</DataArray>'
                % (vtk_type, name, components, encoded))
        cells = []
        for name in ("connectivity", "offsets", "types"):
            vtk_type, encoded = encoded_arrays[name]
            cells.append('<DataArray type="%s" Name="%s" format="binary">%s</DataArray>'
                         % (vtk_type, name, encoded))
        document = '''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid><Piece NumberOfPoints="{points}" NumberOfCells="{cells_count}">
    <FieldData><DataArray type="String" Name="pops_output_manifest" NumberOfTuples="1" format="ascii">{manifest}</DataArray></FieldData>
    <Points><DataArray type="{point_type}" NumberOfComponents="3" format="binary">{point_data}</DataArray></Points>
    <Cells>{cells}</Cells><CellData>{cell_data}</CellData>
  </Piece></UnstructuredGrid>
</VTKFile>
'''.format(points=len(points), cells_count=n_cells, manifest=html.escape(_json(manifest)),
           point_type=point_type, point_data=point_data, cells="".join(cells),
           cell_data="".join(cell_arrays))
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        read_paraview(temporary).require_selection(request)
        return PreparedOutputFile(
            temporary, target, format=self.format, output_identity=identity,
            selection_identity=request.identity, verify=read_paraview)


def read_paraview(path: Any) -> ReopenedOutput:
    root = ET.parse(path).getroot()
    if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("ParaView output is not a VTK UnstructuredGrid")
    manifest_node = root.find(".//FieldData/DataArray[@Name='pops_output_manifest']")
    if manifest_node is None:
        raise ValueError("ParaView output has no PoPS scientific output manifest")
    manifest, identity = _authenticate_manifest(json.loads(manifest_node.text or ""), "paraview-vtu")
    arrays = {}
    points = root.find(".//Points/DataArray")
    if points is None:
        raise ValueError("ParaView output has no point geometry")
    arrays["Points"] = _decode_vtk_array(points)
    for node in root.findall(".//Cells/DataArray") + root.findall(".//CellData/DataArray"):
        arrays[node.attrib["Name"]] = _decode_vtk_array(node)
    if set(arrays) != set(manifest["arrays"]):
        raise ValueError("ParaView arrays differ from its exact manifest")
    for name, evidence in manifest["arrays"].items():
        if array_evidence(arrays[name]) != evidence:
            raise ValueError("ParaView array %r failed verification" % name)
    for record in manifest["datasets"]["fields"].values():
        start, end = record["cell_range"]
        values = arrays[record["name"]][start:end]
        if array_evidence(values) != record["array"]:
            raise ValueError("ParaView selected field failed verification")
    return ReopenedOutput(manifest, arrays, identity)


__all__ = [
    "HDF5Writer", "NPZWriter", "OUTPUT_SCHEMA_VERSION", "OutputPublicationReceipt",
    "ParaViewWriter", "PreparedOutputFile", "ReopenedOutput", "deterministic_target",
    "read_hdf5", "read_npz", "read_paraview",
]
