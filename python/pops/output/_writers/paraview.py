"""Exact SERIAL, ROOT, and PER_RANK VTK UnstructuredGrid backend."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pops.output._writers.common import (
    OutputWriterSession,
    ReopenedOutput,
    _StagedOutputFile,
    _cleanup_staging_authority,
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
from pops.output.data import (
    OutputRequest,
    OutputSnapshot,
    _field_family_identity,
    array_evidence,
)


_VTK_TYPES = {
    "<f8": "Float64",
    "<f4": "Float32",
    "<i8": "Int64",
    "<i4": "Int32",
    "|u1": "UInt8",
}

_VTK_REFINED_CELL = 8
_VTK_FIELD_NAME = re.compile(r"[^A-Za-z0-9_]+")
_VTK_RESERVED_NAMES = frozenset({
    "Points", "TimeValue", "connectivity", "offsets", "types",
    "pops_layout", "pops_level", "pops_coverage", "vtkGhostType",
    "pops_cell_volume",
})


def _field_families(fields: Any) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Group exact level payloads without weakening their logical field identity."""
    grouped: dict[str, list[Any]] = {}
    for field in fields:
        family = _field_family_identity(field.key).token
        grouped.setdefault(family, []).append(field)

    result = []
    for family in sorted(grouped):
        members = tuple(sorted(grouped[family], key=lambda item: item.key.level))
        levels = tuple(item.key.level for item in members)
        if len(levels) != len(set(levels)):
            raise ValueError("ParaView logical field family contains a duplicate level")
        first = members[0]
        invariants = (
            ("centering", first.centering),
            ("units", first.units),
            ("component_names", first.component_names),
            ("dtype", first.array_dtype),
        )
        for name, expected in invariants:
            if any(
                    (item.array_dtype if name == "dtype" else getattr(item, name)) != expected
                    for item in members[1:]):
                raise ValueError(
                    "ParaView logical field family levels disagree on %s" % name)
        result.append((family, members))
    return tuple(result)


def _field_array_names(
    families: tuple[tuple[str, tuple[Any, ...]], ...],
) -> tuple[str, ...]:
    """Use readable block/handle names while retaining collision-proof identities."""
    bases = []
    for _family, members in families:
        reference = members[0].key.reference
        block = getattr(reference, "block_ref", None)
        parts = [getattr(block, "local_id", None), reference.local_id]
        readable = "__".join(str(part) for part in parts if part)
        readable = _VTK_FIELD_NAME.sub("_", readable).strip("_") or "field"
        if readable in _VTK_RESERVED_NAMES:
            readable = "field__" + readable
        bases.append(readable)
    counts = {base: bases.count(base) for base in set(bases)}
    return tuple(
        base if counts[base] == 1 else "%s__%s" % (
            base,
            hashlib.sha256(family.encode("utf-8")).hexdigest()[:8],
        )
        for base, (family, _members) in zip(bases, families, strict=True)
    )


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
        families = _field_families(fields)
        field_array_names = _field_array_names(families)
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
        ghost_types = np.empty(n_cells, dtype="u1")
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
            ghost_types[start:cell] = covered[start:cell] * _VTK_REFINED_CELL
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
            "vtkGhostType": ghost_types,
            "pops_cell_volume": volumes,
            "TimeValue": np.asarray(
                [float.fromhex(snapshot.clock.time_hex)], dtype="<f8"),
        }
        datasets = {"fields": {}, "geometries": {}}
        for ordinal, geometry in enumerate(geometries):
            datasets["geometries"]["%s#%d" % geometry.key] = {
                "layout_ordinal": ordinal,
                "cell_range": list(offsets[geometry.key][:2]),
            }
        component_labels = {}
        for name, (_family, members) in zip(field_array_names, families, strict=True):
            first = members[0]
            component_labels[name] = first.component_names
            components = len(first.component_names)
            shape = (n_cells, components) if components else (n_cells,)
            combined = np.empty(shape, dtype=np.dtype(first.array_dtype))
            written = np.zeros(n_cells, dtype=np.bool_)
            for field in members:
                geometry = snapshot.geometry(field.key)
                valid = emission_masks[geometry.key]
                start, end, ordinal = offsets[geometry.key]
                if np.any(written[start:end]):
                    raise ValueError(
                        "ParaView logical field family maps two levels onto one geometry")
                values = field_values_on_mask(
                    field,
                    valid,
                    require_piece_subset=True,
                )
                combined[start:end, ...] = values
                written[start:end] = True
                datasets["fields"][field.key.identity.token] = {
                    "name": name,
                    "layout_ordinal": ordinal,
                    "cell_range": [start, end],
                    "array": array_evidence(values),
                }
            if not np.all(written):
                raise ValueError(
                    "ParaView logical field family does not cover every emitted geometry")
            combined.setflags(write=False)
            arrays[name] = combined
        evidence = {name: array_evidence(value) for name, value in arrays.items()}
        output_manifest, identity = manifest(
            self.format, snapshot, request, evidence, datasets=datasets)
        encoded_arrays = {name: _vtk_array(value) for name, value in arrays.items()}
        temporary = temporary_path(target)
        point_type, point_data = encoded_arrays["Points"]
        cell_arrays = []
        names = (
            "pops_layout", "pops_level", "pops_coverage", "vtkGhostType",
            "pops_cell_volume",
        ) + field_array_names
        for name in names:
            vtk_type, encoded = encoded_arrays[name]
            components = arrays[name].shape[1] if arrays[name].ndim == 2 else 1
            labels = component_labels.get(name, ())
            component_attributes = "".join(
                ' ComponentName%d="%s"' % (index, html.escape(label, quote=True))
                for index, label in enumerate(labels)
            )
            cell_arrays.append(
                '<DataArray type="%s" Name="%s" NumberOfComponents="%d" '
                '%s format="binary">%s</DataArray>'
                % (vtk_type, name, components, component_attributes, encoded))
        cells = []
        for name in ("connectivity", "offsets", "types"):
            vtk_type, encoded = encoded_arrays[name]
            cells.append(
                '<DataArray type="%s" Name="%s" format="binary">%s</DataArray>'
                % (vtk_type, name, encoded))
        time_type, time_data = encoded_arrays["TimeValue"]
        document = '''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid><FieldData><DataArray type="{time_type}" Name="TimeValue" NumberOfTuples="1" NumberOfComponents="1" format="binary">{time_data}</DataArray></FieldData><Piece NumberOfPoints="{points}" NumberOfCells="{cells_count}">
    <FieldData><DataArray type="String" Name="pops_output_manifest" NumberOfTuples="1" format="ascii">{manifest}</DataArray></FieldData>
    <Points><DataArray type="{point_type}" NumberOfComponents="3" format="binary">{point_data}</DataArray></Points>
    <Cells>{cells}</Cells><CellData>{cell_data}</CellData>
  </Piece></UnstructuredGrid>
</VTKFile>
'''.format(
            points=len(points),
            cells_count=n_cells,
            time_type=time_type,
            time_data=time_data,
            manifest=html.escape(json_text(output_manifest)),
            point_type=point_type,
            point_data=point_data,
            cells="".join(cells),
            cell_data="".join(cell_arrays),
        )
        try:
            with os.fdopen(
                temporary.duplicate(), "w", encoding="utf-8", newline="\n",
            ) as stream:
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            read_paraview(temporary.path).require_selection(request)
            return _StagedOutputFile(
                temporary,
                target,
                format=self.format,
                output_identity=identity,
                selection_identity=request.publication_identity,
                verify=read_paraview,
            )
        except BaseException as error:
            if temporary.is_open:
                try:
                    _cleanup_staging_authority(
                        temporary,
                        replaced_message=(
                            "ParaView staging cleanup refused a replaced temporary at %s"
                            % temporary.path),
                    )
                except BaseException as cleanup_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note("ParaView staging cleanup also failed: %s" % cleanup_error)
            raise


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
    for node in root.findall("./UnstructuredGrid/FieldData/DataArray"):
        name = node.attrib["Name"]
        evidence = output_manifest["arrays"].get(name)
        if evidence is None:
            raise ValueError("ParaView FieldData array %r is unauthenticated" % name)
        arrays[name] = _decode_vtk_array(node, evidence)
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
