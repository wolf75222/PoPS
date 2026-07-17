"""Exact SERIAL, ROOT, and PER_RANK VTK UnstructuredGrid backend."""
from __future__ import annotations

import base64
import html
import json
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pops.output._writers.common import (
    OutputWriterSession,
    ReopenedOutput,
    _StagedOutputFile,
    authenticate_manifest,
    field_values_on_mask,
    json_text,
    manifest,
    selected_geometries,
    temporary_path,
    validate_field_pieces,
    writer_execution_capability,
    writer_session_authority,
)
from pops.output.data import OutputRequest, OutputSnapshot, array_evidence


_VTK_TYPES = {
    "<f8": "Float64",
    "<f4": "Float32",
    "<i8": "Int64",
    "<i4": "Int32",
    "|u1": "UInt8",
}


def _vtk_array(value: Any) -> tuple[str, str]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and struct.pack("=I", 1)[0] != 1):
        array = array.byteswap().newbyteorder("<")
    elif array.dtype.byteorder == "=":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    vtk_type = _VTK_TYPES.get(array.dtype.str)
    if vtk_type is None:
        raise TypeError("ParaView writer does not support dtype %s" % array.dtype)
    raw = array.tobytes(order="C")
    encoded = base64.b64encode(struct.pack("<I", len(raw)) + raw).decode("ascii")
    return vtk_type, encoded


def _decode_vtk_array(node: ET.Element, evidence: Any) -> Any:
    import numpy as np

    types = {value: np.dtype(key) for key, value in _VTK_TYPES.items()}
    dtype = types[node.attrib["type"]]
    raw = base64.b64decode((node.text or "").strip())
    if len(raw) < 4 or struct.unpack("<I", raw[:4])[0] != len(raw) - 4:
        raise ValueError("invalid VTK inline binary array")
    values = np.frombuffer(raw[4:], dtype=dtype).copy()
    components = int(node.attrib.get("NumberOfComponents", "1"))
    if components < 1:
        raise ValueError("VTK array NumberOfComponents must be positive")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("shape"), list):
        raise ValueError("VTK array manifest has no exact shape evidence")
    shape = tuple(evidence["shape"])
    if len(shape) not in (1, 2) or any(
            isinstance(item, bool) or type(item) is not int or item < 0 for item in shape):
        raise ValueError("VTK array manifest shape must be an exact rank-one or rank-two shape")
    expected_components = shape[1] if len(shape) == 2 else 1
    if expected_components != components:
        raise ValueError("VTK array component count differs from its exact manifest shape")
    if values.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("VTK array value count differs from its exact manifest shape")
    return values.reshape(shape)


class ParaViewWriter:
    """Single-file VTK UnstructuredGrid consumer readable directly by ParaView."""

    format = "paraview-vtu"
    extension = ".vtu"

    def __init__(self, mode: Any = None) -> None:
        from pops.output._consumer_contracts import ParallelMode

        if mode is None:
            mode = ParallelMode.SERIAL
        if type(mode) is not ParallelMode:
            raise TypeError("ParaViewWriter mode must be an exact ParallelMode")
        self._mode = mode

    def preflight(self, execution_context: Any) -> dict[str, Any]:
        from pops.output._consumer_contracts import ParallelMode

        if type(self._mode) is not ParallelMode:
            raise RuntimeError("ParaViewWriter preflight requires its resolved format mode")
        return writer_execution_capability(
            execution_context,
            self._mode,
            provider_id="pops.output.paraview-vtu.v1",
        )

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
            raise ValueError("ParaView writer mode differs from its resolved output request")
        if request.parallel_mode not in (
                ParallelMode.SERIAL, ParallelMode.ROOT, ParallelMode.PER_RANK):
            raise ValueError(
                "ParaView VTU supports SERIAL, ROOT, or PER_RANK publication; use HDF5 "
                "for COLLECTIVE output")
        if request.parallel_mode is ParallelMode.SERIAL and communicator is not None:
            raise ValueError("SERIAL ParaView writer session cannot carry a communicator")
        if request.parallel_mode is not ParallelMode.SERIAL and communicator is None:
            raise ValueError("distributed ParaView writer session requires its communicator")
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
            raise ValueError("ParaView target must end in .vtu")
        fields = snapshot.select(request)
        for field in fields:
            validate_field_pieces(
                field,
                snapshot.geometry(field.key),
                complete=request.parallel_mode is not ParallelMode.PER_RANK,
                rank=(None if request.parallel_mode is ParallelMode.ROOT else request.rank),
                size=request.size,
            )
        if any(field.centering != "cell" for field in fields):
            raise NotImplementedError(
                "ParaView VTU currently proves cell-centered arrays only; "
                "no centering substitution")
        geometries = []
        seen = set()
        for field in fields:
            geometry = snapshot.geometry(field.key)
            if geometry.key not in seen:
                seen.add(geometry.key)
                geometries.append(geometry)
        for geometry in selected_geometries(snapshot, request, fields).values():
            if geometry.key not in seen:
                seen.add(geometry.key)
                geometries.append(geometry)
        geometries.sort(key=lambda item: item.key)
        emission_masks = {}
        for geometry in geometries:
            if request.parallel_mode is not ParallelMode.PER_RANK:
                emission_masks[geometry.key] = geometry.valid_cells
                continue
            masks = []
            for field in fields:
                if snapshot.geometry(field.key).key != geometry.key:
                    continue
                mask = np.zeros(geometry.cell_shape, dtype=np.bool_)
                for piece in field.pieces:
                    jlo, ilo = piece.lower
                    jhi, ihi = piece.upper
                    mask[jlo:jhi, ilo:ihi] = True
                masks.append(mask)
            if masks and any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
                raise ValueError(
                    "PER_RANK ParaView fields disagree on their local geometry ownership")
            emission_masks[geometry.key] = (
                masks[0] if masks else np.zeros(geometry.cell_shape, dtype=np.bool_))
        cell_counts = [
            int(np.count_nonzero(emission_masks[geometry.key]))
            for geometry in geometries
        ]
        n_cells = sum(cell_counts)
        points = np.empty((n_cells * 4, 3), dtype="<f8")
        layout_ordinals = np.empty(n_cells, dtype="<i4")
        levels = np.empty(n_cells, dtype="<i4")
        covered = np.empty(n_cells, dtype="u1")
        volumes = np.empty(n_cells, dtype="<f8")
        offsets = {}
        cell = 0
        for ordinal, (geometry, count) in enumerate(
            zip(geometries, cell_counts, strict=True)
        ):
            start = cell
            cell = start + count
            rows, columns = np.nonzero(emission_masks[geometry.key])
            ox, oy = geometry.origin
            dx, dy = geometry.spacing
            x0 = ox + columns * dx
            x1 = ox + (columns + 1) * dx
            y0 = oy + rows * dy
            y1 = oy + (rows + 1) * dy
            cell_points = points[start * 4:cell * 4].reshape((count, 4, 3))
            cell_points[:, 0, 0] = x0
            cell_points[:, 1, 0] = x1
            cell_points[:, 2, 0] = x1
            cell_points[:, 3, 0] = x0
            cell_points[:, 0, 1] = y0
            cell_points[:, 1, 1] = y0
            cell_points[:, 2, 1] = y1
            cell_points[:, 3, 1] = y1
            cell_points[:, :, 2] = 0.0
            layout_ordinals[start:cell] = ordinal
            levels[start:cell] = geometry.level
            covered[start:cell] = geometry.coverage[rows, columns]
            volumes[start:cell] = geometry.cell_volumes[rows, columns]
            offsets[geometry.key] = (start, cell, ordinal)
        arrays = {
            "Points": points,
            "connectivity": np.arange(n_cells * 4, dtype="<i8"),
            "offsets": np.arange(4, n_cells * 4 + 1, 4, dtype="<i8"),
            "types": np.full(n_cells, 9, dtype="u1"),
            "pops_layout": layout_ordinals,
            "pops_level": levels,
            "pops_coverage": covered,
            "pops_cell_volume": volumes,
        }
        datasets = {"fields": {}, "geometries": {}}
        for ordinal, geometry in enumerate(geometries):
            datasets["geometries"]["%s#%d" % geometry.key] = {
                "layout_ordinal": ordinal,
                "cell_range": list(offsets[geometry.key][:2]),
            }
        for index, field in enumerate(fields):
            name = "field_%04d" % index
            geometry = snapshot.geometry(field.key)
            valid = emission_masks[geometry.key]
            components = len(field.component_names) or 1
            start, end, ordinal = offsets[geometry.key]
            values = field_values_on_mask(
                field,
                valid,
                require_piece_subset=True,
            )
            if field.component_names:
                combined = np.zeros((n_cells, components), dtype=values.dtype)
                combined[start:end, :] = values
            else:
                combined = np.zeros(n_cells, dtype=values.dtype)
                combined[start:end] = values
            arrays[name] = combined
            datasets["fields"][field.key.identity.token] = {
                "name": name,
                "layout_ordinal": ordinal,
                "cell_range": [start, end],
                "array": array_evidence(values),
            }
        evidence = {name: array_evidence(value) for name, value in arrays.items()}
        output_manifest, identity = manifest(
            self.format, snapshot, request, evidence, datasets=datasets)
        encoded_arrays = {name: _vtk_array(value) for name, value in arrays.items()}
        temporary = temporary_path(target)
        point_type, point_data = encoded_arrays["Points"]
        cell_arrays = []
        names = (
            "pops_layout", "pops_level", "pops_coverage", "pops_cell_volume",
        ) + tuple("field_%04d" % index for index in range(len(fields)))
        for name in names:
            vtk_type, encoded = encoded_arrays[name]
            components = arrays[name].shape[1] if arrays[name].ndim == 2 else 1
            cell_arrays.append(
                '<DataArray type="%s" Name="%s" NumberOfComponents="%d" '
                'format="binary">%s</DataArray>'
                % (vtk_type, name, components, encoded))
        cells = []
        for name in ("connectivity", "offsets", "types"):
            vtk_type, encoded = encoded_arrays[name]
            cells.append(
                '<DataArray type="%s" Name="%s" format="binary">%s</DataArray>'
                % (vtk_type, name, encoded))
        document = '''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid><Piece NumberOfPoints="{points}" NumberOfCells="{cells_count}">
    <FieldData><DataArray type="String" Name="pops_output_manifest" NumberOfTuples="1" format="ascii">{manifest}</DataArray></FieldData>
    <Points><DataArray type="{point_type}" NumberOfComponents="3" format="binary">{point_data}</DataArray></Points>
    <Cells>{cells}</Cells><CellData>{cell_data}</CellData>
  </Piece></UnstructuredGrid>
</VTKFile>
'''.format(
            points=len(points),
            cells_count=n_cells,
            manifest=html.escape(json_text(output_manifest)),
            point_type=point_type,
            point_data=point_data,
            cells="".join(cells),
            cell_data="".join(cell_arrays),
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        read_paraview(temporary).require_selection(request)
        return _StagedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.publication_identity,
            verify=read_paraview,
        )


def read_paraview(path: Any) -> ReopenedOutput:
    root = ET.parse(path).getroot()
    if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("ParaView output is not a VTK UnstructuredGrid")
    manifest_node = root.find(".//FieldData/DataArray[@Name='pops_output_manifest']")
    if manifest_node is None:
        raise ValueError("ParaView output has no PoPS scientific output manifest")
    output_manifest, identity = authenticate_manifest(
        json.loads(manifest_node.text or ""), "paraview-vtu")
    arrays = {}
    points = root.find(".//Points/DataArray")
    if points is None:
        raise ValueError("ParaView output has no point geometry")
    arrays["Points"] = _decode_vtk_array(points, output_manifest["arrays"]["Points"])
    for node in root.findall(".//Cells/DataArray") + root.findall(".//CellData/DataArray"):
        name = node.attrib["Name"]
        evidence = output_manifest["arrays"].get(name)
        arrays[name] = _decode_vtk_array(node, evidence)
    if set(arrays) != set(output_manifest["arrays"]):
        raise ValueError("ParaView arrays differ from its exact manifest")
    for name, evidence in output_manifest["arrays"].items():
        if array_evidence(arrays[name]) != evidence:
            raise ValueError("ParaView array %r failed verification" % name)
    for record in output_manifest["datasets"]["fields"].values():
        start, end = record["cell_range"]
        values = arrays[record["name"]][start:end]
        if array_evidence(values) != record["array"]:
            raise ValueError("ParaView selected field failed verification")
    return ReopenedOutput(output_manifest, arrays, identity)


__all__ = ["ParaViewWriter", "read_paraview"]
