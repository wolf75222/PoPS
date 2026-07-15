"""Exact serial VTK UnstructuredGrid backend for direct ParaView consumption."""
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
    PreparedOutputFile,
    ReopenedOutput,
    authenticate_manifest,
    json_text,
    manifest,
    selected_geometries,
    temporary_path,
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

    def prepare(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> PreparedOutputFile:
        if communicator is not None:
            raise ValueError("serial ParaView output cannot carry a communicator")
        if request.parallel:
            raise ValueError(
                "ParaView VTU publication is serial; use collective HDF5 for parallel IO")
        import numpy as np

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("ParaView target must end in .vtu")
        fields = snapshot.select(request)
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
        cell_counts = [int(np.count_nonzero(geometry.valid_cells))
                       for geometry in geometries]
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
            rows, columns = np.nonzero(geometry.valid_cells)
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
            dense = field.materialize()
            valid = snapshot.geometry(field.key).valid_cells.ravel()
            components = len(field.component_names) or 1
            start, end, ordinal = offsets[snapshot.geometry(field.key).key]
            if field.component_names and components > 1:
                combined = np.zeros((n_cells, components), dtype=dense.dtype)
                values = dense.reshape((components, -1)).T[valid]
                combined[start:end, :] = values
            else:
                combined = np.zeros(n_cells, dtype=dense.dtype)
                values = dense.reshape(-1)[valid]
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
        return PreparedOutputFile(
            temporary,
            target,
            format=self.format,
            output_identity=identity,
            selection_identity=request.identity,
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
    arrays["Points"] = _decode_vtk_array(points)
    for node in root.findall(".//Cells/DataArray") + root.findall(".//CellData/DataArray"):
        arrays[node.attrib["Name"]] = _decode_vtk_array(node)
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
