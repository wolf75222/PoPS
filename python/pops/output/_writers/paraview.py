"""Exact SERIAL, ROOT, and PER_RANK VTK UnstructuredGrid backend."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

from pops._geometry_contracts import (
    CARTESIAN_1D_COORDINATES,
    CARTESIAN_2D_COORDINATES,
    CARTESIAN_3D_COORDINATES,
    POLAR_ANNULUS_2D_COORDINATES,
)
from pops.output._writers.common import (
    OutputPublicationReceipt,
    ReopenedOutput,
    _FACE_CENTERINGS,
    _StagedOutputFile,
    _cleanup_staging_authority,
    _face_array_axis,
    _series_selection_authority,
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
    EMBEDDED_BOUNDARY_ARRAY_NAMES,
    EmbeddedBoundaryPayload,
    OutputRequest,
    OutputSnapshot,
    _centering_shape,
    _field_family_identity,
    array_evidence,
)
from pops.identity import Identity, make_identity
from pops._native_collectives import allgather_value, gather_bytes


_VTK_TYPES = {
    "<f8": "Float64",
    "<f4": "Float32",
    "<i8": "Int64",
    "<i4": "Int32",
    "|u1": "UInt8",
}

_VTK_DUPLICATE_CELL = 1
_VTK_REFINED_CELL = 8
_VTK_VERTEX = 1
_VTK_CELL = {1: (3, 2), 2: (9, 4), 3: (12, 8)}
# Face-centered MAC locations are (Dim-1)-cells: vertices in 1D, lines in 2D, quads in 3D.
_VTK_FACE = {1: (_VTK_VERTEX, 1), 2: (3, 2), 3: (9, 4)}
_VTK_COMPRESSION_BLOCK_SIZE = 32768
_VTK_FIELD_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_VTK_RESERVED_ARRAY_NAMES = frozenset({
    "Points", "connectivity", "offsets", "types", "TimeValue",
    "pops_layout", "pops_level", "pops_coverage", "vtkGhostType",
    "pops_cell_volume", "pops_output_manifest", *EMBEDDED_BOUNDARY_ARRAY_NAMES,
})


@dataclass(frozen=True, slots=True)
class ReopenedParaViewIndex:
    """Authenticated standard ParaView index and its relative component paths."""

    kind: str
    manifest: dict[str, Any]
    paths: tuple[Path, ...]
    output_identity: Identity

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the collection members using the generic series vocabulary."""
        return self.paths

    @property
    def times(self) -> tuple[float, ...]:
        """Return the authenticated physical-time axis of a PVD collection."""
        if self.kind != "pvd":
            raise RuntimeError("only a ParaView PVD index has a temporal axis")
        return _pvd_time_values(self.manifest.get("entries", ()))

    @property
    def latest(self) -> ReopenedOutput | ReopenedParaViewIndex | ReopenedParaViewMultiBlock:
        """Reopen the latest authenticated member of a PVD collection."""
        if self.kind != "pvd":
            raise RuntimeError("only a ParaView PVD index has a latest sample")
        if not self.paths:
            raise RuntimeError("ParaView PVD collection has no samples")
        return read_paraview(self.paths[-1])

    def verify(self) -> ReopenedParaViewIndex:
        """Reopen every referenced component while retaining no field arrays."""
        for path in self.paths:
            read_paraview(path)
        return self


@dataclass(frozen=True, slots=True)
class ReopenedParaViewMultiBlock:
    """Authenticated VTK MultiBlock of honest Cartesian cell/face topologies."""

    kind: str
    manifest: dict[str, Any]
    blocks: tuple[tuple[str, ReopenedOutput], ...]
    output_identity: Identity
    path: Path

    def require_selection(self, request: OutputRequest) -> ReopenedParaViewMultiBlock:
        recorded = self.manifest.get("snapshot", {}).get("selection")
        if recorded != request.publication_data():
            raise ValueError("reopened output selection differs from the requested selection")
        return self

    def block(self, name: str) -> ReopenedOutput:
        matches = [item for item in self.blocks if item[0] == name]
        if len(matches) != 1:
            raise KeyError("ParaView MultiBlock has no unique topology %r" % name)
        return matches[0][1]


def _exception_text(error: BaseException) -> str:
    return "%s: %s" % (type(error).__name__, error)


def _subprocess_detail(error: BaseException) -> str:
    for name in ("stderr", "stdout"):
        value = getattr(error, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip().splitlines()[-1][:1000]
    return _exception_text(error)


def _safe_name(value: str) -> str:
    import re

    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not result:
        raise ValueError("ParaView catalogue name has no safe characters")
    return result[:60]


def _relative_component(parent: Path, component: Path) -> str:
    resolved_parent = parent.expanduser().resolve()
    resolved = component.expanduser().resolve()
    try:
        relative = resolved.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError("ParaView companion files must share one output directory") from exc
    if len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ValueError("ParaView companion reference must be one relative filename")
    return relative.as_posix()


def _index_attributes(domain: str, payload: dict[str, Any]) -> tuple[str, str, Identity]:
    identity = make_identity(domain, payload)
    encoded = base64.urlsafe_b64encode(json_text(payload).encode("utf-8")).decode("ascii")
    return encoded, identity.token, identity


def _read_index_attributes(
    root: ET.Element, domain: str,
) -> tuple[dict[str, Any], Identity]:
    encoded = root.attrib.get("pops_manifest")
    supplied = root.attrib.get("pops_identity")
    if not encoded or not supplied:
        raise ValueError("ParaView index has no authenticated PoPS manifest")
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("ParaView index manifest is not canonical JSON data") from exc
    expected = make_identity(domain, payload)
    try:
        identity = Identity.from_token(supplied)
    except (TypeError, ValueError) as exc:
        raise ValueError("ParaView index identity is invalid") from exc
    if identity != expected:
        raise ValueError("ParaView index identity mismatch")
    return payload, identity


def _component_path(index: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or len(candidate.parts) != 1 \
            or candidate.name in {"", ".", ".."}:
        raise ValueError("ParaView index contains a non-local component path")
    return index.parent / candidate


def _pvd_time_values(entries: Any) -> tuple[float, ...]:
    """Authenticate one strictly increasing binary64 time axis."""
    import math

    values = []
    for row in entries:
        if not isinstance(row, dict) or not isinstance(row.get("time_hex"), str):
            raise ValueError("ParaView PVD entry has no canonical physical time")
        try:
            value = float.fromhex(row["time_hex"])
        except (OverflowError, ValueError):
            raise ValueError("ParaView PVD entry has no canonical physical time") from None
        if not math.isfinite(value) or value.hex() != row["time_hex"]:
            raise ValueError("ParaView PVD entry has no canonical physical time")
        values.append(value)
    if any(values[index] <= values[index - 1] for index in range(1, len(values))):
        raise ValueError("ParaView PVD physical times are not strictly increasing")
    return tuple(values)


@lru_cache(maxsize=8)
def _pvpython_capability(requested: str | None) -> tuple[str, str]:
    candidate = shutil.which("pvpython" if requested is None else requested)
    if candidate is None and requested is not None:
        path = Path(requested).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            candidate = str(path.resolve())
    if candidate is None:
        raise RuntimeError(
            "ParaView MaterializedPVSM requires an executable pvpython; configure "
            "MaterializedPVSM(pvpython=...)"
        )
    try:
        completed = subprocess.run(
            [candidate, "--no-mpi", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("configured pvpython failed its version preflight") from exc
    version = (completed.stdout or completed.stderr).strip().splitlines()
    if not version:
        raise RuntimeError("configured pvpython returned no version evidence")
    return str(Path(candidate).resolve()), version[0]


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


def _field_display_names(
    families: tuple[tuple[str, tuple[Any, ...]], ...],
) -> tuple[str, ...]:
    """Prefer declaration names and disambiguate them without losing exact identity.

    Python assignment names do not survive authoring, but every selected Handle retains the name
    declared by the user.  Homonymous declarations in different blocks first gain their block name;
    any remaining collision receives a short digest of the authenticated logical-field family.
    """
    def readable(value: Any) -> str:
        result = _VTK_FIELD_NAME.sub("_", str(value)).strip("_.-")
        return result or "field"

    preferred = [
        readable(members[0].key.reference.local_id)
        for _family, members in families
    ]
    counts = {name: preferred.count(name) for name in set(preferred)}
    candidates = []
    for name, (_family, members) in zip(preferred, families, strict=True):
        reference = members[0].key.reference
        candidate = name
        block = getattr(reference, "block_ref", None)
        block_name = None if block is None else getattr(block, "local_id", None)
        if block_name is None:
            block_names = [
                getattr(segment, "name", None)
                for segment in getattr(
                    getattr(reference, "owner_path", None), "nodes", ())
                if getattr(getattr(segment, "kind", None), "value", None) == "block"
                and getattr(segment, "name", None)
            ]
            block_name = None if not block_names else block_names[-1]
        if counts[name] > 1 and block_name is not None:
            candidate = "%s.%s" % (readable(block_name), name)
        if candidate in _VTK_RESERVED_ARRAY_NAMES:
            candidate = "field.%s" % candidate
        candidates.append(candidate)
    candidate_counts = {name: candidates.count(name) for name in set(candidates)}
    result = []
    for candidate, (family, _members) in zip(candidates, families, strict=True):
        if candidate_counts[candidate] > 1:
            candidate = "%s__%s" % (
                candidate,
                hashlib.sha256(family.encode("utf-8")).hexdigest()[:12],
            )
        result.append(candidate)
    if len(result) != len(set(result)):
        raise RuntimeError("ParaView display-name disambiguation is not injective")
    return tuple(result)


def _field_array_names(
    families: tuple[tuple[str, tuple[Any, ...]], ...],
) -> tuple[str, ...]:
    """Backward-compatible internal name for the display-name resolver."""
    return _field_display_names(families)


def _physical_point_coordinates(
    geometry: Any, logical_indices: tuple[Any, ...],
) -> Any:
    """Map rank-1/2/3 logical vertices to VTK's three Cartesian coordinates."""
    import numpy as np

    dimension = geometry.spatial_rank
    if len(logical_indices) != dimension:
        raise ValueError("ParaView point indices differ from geometry spatial rank")
    cartesian = {
        1: CARTESIAN_1D_COORDINATES,
        2: CARTESIAN_2D_COORDINATES,
        3: CARTESIAN_3D_COORDINATES,
    }
    points = np.zeros((len(logical_indices[0]), 3), dtype="<f8")
    if geometry.coordinate_system == cartesian[dimension]:
        for coordinate_axis in range(dimension):
            array_axis = dimension - 1 - coordinate_axis
            points[:, coordinate_axis] = (
                geometry.origin[coordinate_axis]
                + logical_indices[array_axis] * geometry.spacing[coordinate_axis]
            )
        return points
    if geometry.coordinate_system == POLAR_ANNULUS_2D_COORDINATES:
        if dimension != 2:
            raise ValueError("polar-annulus coordinates require spatial rank two")
        radius = geometry.origin[0] + logical_indices[1] * geometry.spacing[0]
        angle = geometry.origin[1] + logical_indices[0] * geometry.spacing[1]
        points[:, 0] = radius * np.cos(angle)
        points[:, 1] = radius * np.sin(angle)
        return points
    raise NotImplementedError(
        "ParaView point mapping does not support coordinate system %s"
        % geometry.coordinate_system
    )


def _spatial_slices(lower: Any, upper: Any) -> tuple[slice, ...]:
    return tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))


def _embedded_values_on_mask(
    sidecar: EmbeddedBoundaryPayload,
    name: str,
    mask: Any,
    *,
    require_piece_subset: bool,
) -> Any:
    """Project one exact scalar EB sidecar onto a writer-owned emission mask."""
    import numpy as np

    selected = np.asarray(mask, dtype=np.bool_)
    if selected.shape != sidecar.global_shape:
        raise ValueError("embedded-boundary emission mask differs from its geometry")
    dense = np.empty(sidecar.global_shape, dtype=np.float64)
    written = np.zeros(sidecar.global_shape, dtype=np.bool_)
    for piece in sidecar.pieces(name):
        spatial = _spatial_slices(piece.lower, piece.upper)
        if np.any(written[spatial]):
            raise ValueError("embedded-boundary output pieces overlap")
        dense[spatial] = piece.values[0]
        written[spatial] = True
    if np.any(selected & ~written):
        raise ValueError("embedded-boundary sidecar does not cover every emitted geometry cell")
    if require_piece_subset and np.any(written & ~selected):
        raise ValueError(
            "embedded-boundary sidecar ownership differs from physical field ownership"
        )
    return np.ascontiguousarray(dense[selected])


def _point_mask(cell_mask: Any) -> Any:
    import numpy as np

    cells = np.asarray(cell_mask, dtype=np.bool_)
    if cells.ndim not in (1, 2, 3):
        raise ValueError("VTK structured-cell masks require spatial rank 1, 2, or 3")
    points = np.zeros(tuple(extent + 1 for extent in cells.shape), dtype=np.bool_)
    for corner in product((0, 1), repeat=cells.ndim):
        target = tuple(
            slice(offset, offset + extent)
            for offset, extent in zip(corner, cells.shape, strict=True))
        points[target] |= cells
    return points


def _cell_corner_offsets(dimension: int) -> tuple[tuple[int, ...], ...]:
    if dimension not in _VTK_CELL:
        raise ValueError("VTK cells require spatial rank 1, 2, or 3")

    # VTK orders a line directly, a quadrilateral by a binary-reflected Gray
    # cycle, and a hexahedron as two equally oriented quadrilateral faces.
    # Express that tensor-product rule once instead of maintaining one topology
    # algorithm per spatial rank.
    face_rank = min(dimension, 2)
    prefix_rank = dimension - face_rank
    face = tuple(
        tuple(
            ((ordinal ^ (ordinal >> 1)) >> (face_rank - axis - 1)) & 1
            for axis in range(face_rank)
        )
        for ordinal in range(1 << face_rank)
    )
    return tuple(
        prefix + corner
        for prefix in product((0, 1), repeat=prefix_rank)
        for corner in face
    )


def _face_corner_offsets(
    dimension: int, array_axis: int,
) -> tuple[tuple[int, ...], ...]:
    """VTK-wound corners of one Cartesian face whose normal is ``array_axis``."""
    if dimension not in _VTK_FACE:
        raise ValueError("VTK faces require spatial rank 1, 2, or 3")
    if array_axis not in range(dimension):
        raise ValueError("face normal array axis is outside the spatial rank")
    tangent_axes = tuple(axis for axis in range(dimension) if axis != array_axis)
    tangent_rank = len(tangent_axes)
    if tangent_rank == 0:
        return ((0,) * dimension,)
    corners = []
    for ordinal in range(1 << tangent_rank):
        gray = ordinal ^ (ordinal >> 1)
        offset = [0] * dimension
        for tangent_index, axis in enumerate(tangent_axes):
            offset[axis] = (gray >> (tangent_rank - tangent_index - 1)) & 1
        corners.append(tuple(offset))
    return tuple(corners)


def _staggered_face_mask(cell_mask: Any, array_axis: int) -> Any:
    """Unique faces incident on a cell mask, without duplicating shared interfaces."""
    import numpy as np

    cells = np.asarray(cell_mask, dtype=np.bool_)
    if array_axis not in range(cells.ndim):
        raise ValueError("face normal array axis is outside the spatial rank")
    face_shape = list(cells.shape)
    face_shape[array_axis] += 1
    faces = np.zeros(tuple(face_shape), dtype=np.bool_)
    low = [slice(None)] * cells.ndim
    high = [slice(None)] * cells.ndim
    low[array_axis] = slice(0, cells.shape[array_axis])
    high[array_axis] = slice(1, cells.shape[array_axis] + 1)
    faces[tuple(low)] |= cells
    faces[tuple(high)] |= cells
    return faces


def _face_point_mask(face_mask: Any, array_axis: int) -> Any:
    """Vertices of a staggered face mask.  The normal axis is not dilated."""
    import numpy as np

    faces = np.asarray(face_mask, dtype=np.bool_)
    if array_axis not in range(faces.ndim):
        raise ValueError("face normal array axis is outside the spatial rank")
    point_shape = [
        extent + (0 if axis == array_axis else 1)
        for axis, extent in enumerate(faces.shape)
    ]
    points = np.zeros(tuple(point_shape), dtype=np.bool_)
    tangent_axes = tuple(axis for axis in range(faces.ndim) if axis != array_axis)
    for corner in product((0, 1), repeat=len(tangent_axes)):
        target = [slice(None)] * faces.ndim
        target[array_axis] = slice(0, faces.shape[array_axis])
        for offset, axis in zip(corner, tangent_axes, strict=True):
            target[axis] = slice(offset, offset + faces.shape[axis])
        points[tuple(target)] |= faces
    return points


_PRIMAL_CENTERINGS = frozenset({"cell", "node"})
_FACE_CENTERING_ORDER = ("face_x", "face_y", "face_z")


def _fields_for_topology(fields: Any, topology: str) -> tuple[Any, ...]:
    """Return the fields that honestly live on one Cartesian topology."""

    if topology == "cell":
        return tuple(field for field in fields if field.centering in _PRIMAL_CENTERINGS)
    if topology not in _FACE_CENTERINGS:
        raise ValueError("Cartesian publication topology %r is not cell or a face axis" % topology)
    return tuple(field for field in fields if field.centering == topology)


def _publication_topologies(
    fields: Any,
    *,
    embedded: bool,
    spatial_rank: int,
) -> tuple[str, ...]:
    """Return the honest VTK topologies required by one Cartesian snapshot.

    Cell and nodal fields share the primal LINE/QUAD/HEX mesh.  Each face axis is a
    distinct VERTEX/LINE/QUAD mesh because staggered counts differ.  Mixed publications
    therefore use sibling MultiBlock datasets rather than padding one CellData array.
    An embedded-boundary sidecar describes primal cells, so a face-only selection still
    emits a cell block for those arrays and never copies them onto the face mesh.
    """
    centerings = {field.centering for field in fields}
    unknown = centerings - _PRIMAL_CENTERINGS - _FACE_CENTERINGS
    if unknown:
        raise NotImplementedError(
            "Cartesian output proves cell-centered, nodal, and face-centered fields; "
            "other centerings require a distinct topology: %s" % (tuple(sorted(unknown)),)
        )
    faces = tuple(name for name in _FACE_CENTERING_ORDER if name in centerings)
    for face in faces:
        _face_array_axis(spatial_rank, face)
    primal = bool(centerings & _PRIMAL_CENTERINGS) or (embedded and bool(faces))
    topologies = (("cell",) if primal else ()) + faces
    if not topologies:
        raise ValueError("Cartesian output has no cell, node, or face topology to publish")
    return topologies


def _resolved_face_centering(fields: Any) -> str | None:
    """Return the single face centering of a homogeneous snapshot, or None for a primal mesh.

    Mixed cell/face or multi-axis face selections are sibling MultiBlock topologies, not
    one padded UnstructuredGrid.  Callers that need the full set use
    :func:`_publication_topologies`.
    """
    centerings = {field.centering for field in fields}
    faces = tuple(name for name in _FACE_CENTERING_ORDER if name in centerings)
    if faces and (centerings & _PRIMAL_CENTERINGS or len(faces) > 1):
        return None
    return None if not faces else faces[0]


def _topology_request(request: OutputRequest, fields: Any) -> OutputRequest | None:
    """Build a subset request for one topology, or None when only geometry/EB remains."""

    keys = tuple(field.key for field in fields)
    if not keys:
        return None
    return OutputRequest(
        request.consumer_id,
        keys,
        request.parallel_mode,
        rank=request.rank,
        size=request.size,
        diagnostics=request.diagnostics,
    )


def _base64_size(byte_count: int) -> int:
    return ((byte_count + 2) // 3) * 4


def _vtk_array(value: Any, compression: int | None) -> tuple[str, str]:
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
    if compression is None:
        if len(raw) >= 2 ** 32:
            raise OverflowError("VTK UInt32 inline array framing is limited to 4 GiB")
        encoded = base64.b64encode(struct.pack("<I", len(raw)) + raw).decode("ascii")
        return vtk_type, encoded
    full_blocks, remainder = divmod(len(raw), _VTK_COMPRESSION_BLOCK_SIZE)
    number_of_blocks = full_blocks + (1 if remainder else 0)
    blocks = [
        raw[index:index + _VTK_COMPRESSION_BLOCK_SIZE]
        for index in range(0, len(raw), _VTK_COMPRESSION_BLOCK_SIZE)
    ]
    compressed = [zlib.compress(block, compression) for block in blocks]
    sizes = [len(block) for block in compressed]
    if any(size >= 2 ** 32 for size in sizes):
        raise OverflowError("one compressed VTK block exceeds UInt32 framing")
    header = struct.pack(
        "<%dI" % (3 + number_of_blocks),
        number_of_blocks,
        _VTK_COMPRESSION_BLOCK_SIZE,
        remainder,
        *sizes,
    )
    # VTK base64-encodes the compression header separately from the block payload.
    encoded = (
        base64.b64encode(header).decode("ascii")
        + base64.b64encode(b"".join(compressed)).decode("ascii")
    )
    return vtk_type, encoded


def _decode_vtk_array(
    node: ET.Element,
    evidence: Any,
    *,
    compressor: str | None,
) -> Any:
    import numpy as np

    types = {value: np.dtype(key) for key, value in _VTK_TYPES.items()}
    dtype = types[node.attrib["type"]]
    encoded = (node.text or "").strip()
    if compressor is None:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) < 4 or struct.unpack("<I", raw[:4])[0] != len(raw) - 4:
            raise ValueError("invalid VTK inline binary array")
        payload = raw[4:]
    else:
        if compressor != "vtkZLibDataCompressor":
            raise ValueError("unsupported VTK compressor %r" % compressor)
        first = base64.b64decode(encoded[:8], validate=True)
        if len(first) < 4:
            raise ValueError("compressed VTK array has no complete block header")
        number_of_blocks = struct.unpack("<I", first[:4])[0]
        header_size = 4 * (3 + number_of_blocks)
        header_chars = _base64_size(header_size)
        header = base64.b64decode(encoded[:header_chars], validate=True)
        if len(header) != header_size:
            raise ValueError("compressed VTK array has a truncated block header")
        words = struct.unpack("<%dI" % (3 + number_of_blocks), header)
        _blocks, block_size, last_size, *compressed_sizes = words
        if block_size < 1 or block_size % 8:
            raise ValueError("compressed VTK array has an invalid block size")
        compressed = base64.b64decode(encoded[header_chars:], validate=True)
        if sum(compressed_sizes) != len(compressed):
            raise ValueError("compressed VTK block sizes differ from their payload")
        pieces = []
        offset = 0
        for index, compressed_size in enumerate(compressed_sizes):
            block = zlib.decompress(compressed[offset:offset + compressed_size])
            offset += compressed_size
            expected = last_size if index == number_of_blocks - 1 and last_size else block_size
            if len(block) != expected:
                raise ValueError("compressed VTK block has an invalid uncompressed size")
            pieces.append(block)
        payload = b"".join(pieces)
    values = np.frombuffer(payload, dtype=dtype).copy()
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


def _vtu_schema(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    points = root.find("./UnstructuredGrid/Piece/Points/DataArray")
    if points is None:
        raise ValueError("VTU schema has no Points array")
    point = {
        "type": points.attrib.get("type"),
        "components": int(points.attrib.get("NumberOfComponents", "1")),
    }
    def arrays_at(xpath: str, association: str) -> list[dict[str, Any]]:
        arrays = []
        for node in root.findall(xpath):
            arrays.append({
                "name": node.attrib.get("Name"),
                "type": node.attrib.get("type"),
                "components": int(node.attrib.get("NumberOfComponents", "1")),
                "component_names": [
                    node.attrib["ComponentName%d" % index]
                    for index in range(int(node.attrib.get("NumberOfComponents", "1")))
                    if "ComponentName%d" % index in node.attrib
                ],
            })
        if any(
                not row["name"] or row["type"] not in set(_VTK_TYPES.values())
                or row["components"] < 1 for row in arrays):
            raise ValueError("VTU schema contains an invalid %s declaration" % association)
        return arrays

    point_arrays = arrays_at(
        "./UnstructuredGrid/Piece/PointData/DataArray", "PointData")
    cell_arrays = arrays_at(
        "./UnstructuredGrid/Piece/CellData/DataArray", "CellData")
    return {
        "points": point,
        "point_arrays": point_arrays,
        "cell_arrays": cell_arrays,
    }


def _unique_schema_arrays(rows: list[dict[str, Any]], *, where: str) -> list[dict[str, Any]]:
    """Union VTK array declarations that share a name only when their schema matches."""

    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["name"]
        previous = by_name.get(name)
        if previous is None:
            by_name[name] = row
            continue
        if previous != row:
            raise ValueError("ParaView MultiBlock %s array %r has conflicting schemas" % (where, name))
    return list(by_name.values())


def _schema_from_vtu_paths(
    paths: tuple[Path, ...],
    *,
    block_names: tuple[str, ...],
) -> dict[str, Any]:
    if len(paths) != len(block_names):
        raise ValueError("ParaView MultiBlock schema files disagree on topology count")
    point_arrays: list[dict[str, Any]] = []
    cell_arrays: list[dict[str, Any]] = []
    points = None
    for path in paths:
        schema = _vtu_schema(path)
        if points is None:
            points = schema["points"]
        elif points != schema["points"]:
            raise ValueError("ParaView MultiBlock blocks disagree on their Points schema")
        point_arrays.extend(schema["point_arrays"])
        cell_arrays.extend(schema["cell_arrays"])
    if points is None:
        raise ValueError("ParaView MultiBlock has no topology blocks")
    return {
        "points": points,
        "point_arrays": _unique_schema_arrays(point_arrays, where="PointData"),
        "cell_arrays": _unique_schema_arrays(cell_arrays, where="CellData"),
        "blocks": list(block_names),
    }


def _vtm_schema(path: Path) -> dict[str, Any]:
    reopened = _read_vtm(path, verify_components=True)
    return _schema_from_vtu_paths(
        tuple(_component_path(path, row["file"]) for row in reopened.manifest["blocks"]),
        block_names=tuple(row["name"] for row in reopened.manifest["blocks"]),
    )


def _publication_schema(
    path: Path,
    *,
    companion_paths: tuple[Path, ...] = (),
    block_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if root.tag == "VTKFile" and root.attrib.get("type") == "vtkMultiBlockDataSet":
        if companion_paths:
            return _schema_from_vtu_paths(companion_paths, block_names=block_names)
        return _vtm_schema(path)
    return _vtu_schema(path)


def _stage_text_file(
    target: Path,
    document: str | bytes,
    *,
    format_name: str,
    output_identity: Identity,
    selection_identity: Identity,
    verify: Any,
) -> _StagedOutputFile:
    temporary = temporary_path(target)
    try:
        payload = document.encode("utf-8") if isinstance(document, str) else bytes(document)
        with os.fdopen(temporary.duplicate(), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        verified = verify(temporary.path)
        if getattr(verified, "output_identity", None) != output_identity:
            raise ValueError("staged ParaView companion identity differs after reopen")
        return _StagedOutputFile(
            temporary,
            target,
            format=format_name,
            output_identity=output_identity,
            selection_identity=selection_identity,
            verify=verify,
        )
    except BaseException as error:
        if temporary.is_open:
            try:
                _cleanup_staging_authority(
                    temporary,
                    replaced_message=(
                        "ParaView companion cleanup refused a replaced temporary at %s"
                        % temporary.path),
                )
            except BaseException as cleanup_error:
                add_note = getattr(error, "add_note", None)
                if callable(add_note):
                    add_note("ParaView companion cleanup also failed: %s" % cleanup_error)
        raise


def _stage_portable_state(
    pvd: _StagedOutputFile,
    request: OutputRequest,
    schema: dict[str, Any],
    preset: Any,
) -> tuple[_StagedOutputFile, _StagedOutputFile]:
    """Stage one relocatable recipe/script pair without importing ParaView."""

    from pops.output.paraview_state import build_portable_paraview_state

    manifest_target = pvd.target.with_suffix(".view.json")
    script_target = manifest_target.with_suffix(".py")
    resolved = _resolved_preset_data(schema, preset)
    presentation = {
        key: value for key, value in resolved.items() if key != "color_association"
    }
    documents = build_portable_paraview_state(
        pvd_file=pvd.target.name,
        pvd_identity=pvd.output_identity.token,
        presentation=presentation,
        cell_arrays=schema["cell_arrays"],
        point_arrays=schema.get("point_arrays", ()),
        manifest_file=manifest_target.name,
        script_file=script_target.name,
    )
    script_identity = make_identity("paraview-portable-script", {
        "state_identity": documents.identity.token,
        "sha256": hashlib.sha256(documents.script).hexdigest(),
    })

    def verify_exact(expected: bytes, identity: Identity, kind: str) -> Any:
        def verify(path: Path) -> ReopenedParaViewIndex:
            if path.read_bytes() != expected:
                raise ValueError("staged %s differs from its deterministic bytes" % kind)
            return ReopenedParaViewIndex(kind, {}, (), identity)

        return verify

    script = _stage_text_file(
        script_target,
        documents.script,
        format_name="paraview-portable-script",
        output_identity=script_identity,
        selection_identity=request.publication_identity,
        verify=verify_exact(documents.script, script_identity, "portable ParaView script"),
    )
    try:
        manifest = _stage_text_file(
            manifest_target,
            documents.manifest,
            format_name="paraview-portable-state",
            output_identity=documents.identity,
            selection_identity=request.publication_identity,
            verify=verify_exact(
                documents.manifest, documents.identity, "portable ParaView manifest"),
        )
    except BaseException:
        script.discard()
        raise
    return script, manifest


def _read_pvtu(path: Any, *, verify_components: bool = True) -> ReopenedParaViewIndex:
    index = Path(path)
    root = ET.parse(index).getroot()
    if root.tag != "VTKFile" or root.attrib.get("type") != "PUnstructuredGrid":
        raise ValueError("ParaView parallel output is not a PUnstructuredGrid")
    manifest_data, identity = _read_index_attributes(root, "paraview-pvtu")
    if manifest_data.get("schema_version") != 1:
        raise ValueError("ParaView PVTU manifest has an unsupported schema")
    grid = root.find("./PUnstructuredGrid")
    if grid is None or grid.attrib.get("GhostLevel") != "0":
        raise ValueError("ParaView PVTU has an invalid parallel grid declaration")
    xml_pieces = [node.attrib.get("Source") for node in grid.findall("./Piece")]
    expected_pieces = [row["file"] for row in manifest_data.get("pieces", ())]
    if xml_pieces != expected_pieces:
        raise ValueError("ParaView PVTU pieces differ from its manifest")
    points = grid.find("./PPoints/PDataArray")
    point_schema = manifest_data.get("schema", {}).get("points")
    if points is None or {
        "type": points.attrib.get("type"),
        "components": int(points.attrib.get("NumberOfComponents", "1")),
    } != point_schema:
        raise ValueError("ParaView PVTU point schema differs from its manifest")
    def parallel_arrays(xpath: str) -> list[dict[str, Any]]:
        result = []
        for node in grid.findall(xpath):
            components = int(node.attrib.get("NumberOfComponents", "1"))
            result.append({
                "name": node.attrib.get("Name"),
                "type": node.attrib.get("type"),
                "components": components,
                "component_names": [
                    node.attrib["ComponentName%d" % index]
                    for index in range(components)
                    if "ComponentName%d" % index in node.attrib
                ],
            })
        return result

    xml_points = parallel_arrays("./PPointData/PDataArray")
    if xml_points != manifest_data.get("schema", {}).get("point_arrays"):
        raise ValueError("ParaView PVTU point-data schema differs from its manifest")
    xml_cells = parallel_arrays("./PCellData/PDataArray")
    if xml_cells != manifest_data.get("schema", {}).get("cell_arrays"):
        raise ValueError("ParaView PVTU cell schema differs from its manifest")
    paths = tuple(_component_path(index, relative) for relative in expected_pieces)
    if verify_components:
        for record, component in zip(manifest_data["pieces"], paths, strict=True):
            reopened = read_paraview(component)
            if reopened.output_identity.token != record["output_identity"]:
                raise ValueError("ParaView PVTU component identity mismatch")
            if _vtu_schema(component) != manifest_data["schema"]:
                raise ValueError("ParaView PVTU component schema mismatch")
    return ReopenedParaViewIndex("pvtu", manifest_data, paths, identity)


def _read_pvd(path: Any, *, verify_components: bool = True) -> ReopenedParaViewIndex:
    index = Path(path)
    root = ET.parse(index).getroot()
    if root.tag != "VTKFile" or root.attrib.get("type") != "Collection":
        raise ValueError("ParaView temporal output is not a Collection")
    manifest_data, identity = _read_index_attributes(root, "paraview-pvd")
    if manifest_data.get("schema_version") != 1:
        raise ValueError("ParaView PVD manifest has an unsupported schema")
    collection = root.find("./Collection")
    if collection is None:
        raise ValueError("ParaView PVD has no Collection")
    xml_entries = []
    for node in collection.findall("./DataSet"):
        xml_entries.append({
            "time": node.attrib.get("timestep"),
            "group": node.attrib.get("group"),
            "part": node.attrib.get("part"),
            "file": node.attrib.get("file"),
        })
    entries = manifest_data.get("entries", ())
    expected_xml = [{
        "time": format(float.fromhex(row["time_hex"]), ".17g"),
        "group": "",
        "part": "0",
        "file": row["file"],
    } for row in entries]
    if xml_entries != expected_xml:
        raise ValueError("ParaView PVD entries differ from its manifest")
    steps = [row["macro_step"] for row in entries]
    if steps != sorted(set(steps)):
        raise ValueError("ParaView PVD macro steps are not strictly increasing")
    _pvd_time_values(entries)
    paths = tuple(_component_path(index, row["file"]) for row in entries)
    if verify_components:
        for record, component in zip(entries, paths, strict=True):
            if component.suffix == ".pvtu":
                reopened = _read_pvtu(component, verify_components=True)
                time_hex = reopened.manifest["clock"]["time"]
            elif component.suffix == ".vtu":
                reopened = read_paraview(component)
                time_hex = reopened.manifest["snapshot"]["clock"]["time"]
            elif component.suffix == ".vtm":
                reopened = _read_vtm(component, verify_components=True)
                time_hex = reopened.manifest["snapshot"]["clock"]["time"]
            else:
                raise ValueError("ParaView PVD references an unsupported component")
            if reopened.output_identity.token != record["output_identity"]:
                raise ValueError("ParaView PVD component identity mismatch")
            if time_hex != record["time_hex"]:
                raise ValueError("ParaView PVD timestep differs from its component")
    return ReopenedParaViewIndex("pvd", manifest_data, paths, identity)


def _read_vtm(path: Any, *, verify_components: bool = True) -> ReopenedParaViewMultiBlock:
    index = Path(path)
    root = ET.parse(index).getroot()
    if root.tag != "VTKFile" or root.attrib.get("type") != "vtkMultiBlockDataSet":
        raise ValueError("ParaView MultiBlock output is not a vtkMultiBlockDataSet")
    manifest_data, identity = _read_index_attributes(root, "paraview-vtm")
    if manifest_data.get("schema_version") != 1:
        raise ValueError("ParaView MultiBlock manifest has an unsupported schema")
    blocks = manifest_data.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("ParaView MultiBlock has no topology blocks")
    xml_blocks = []
    dataset = root.find("./vtkMultiBlockDataSet")
    if dataset is None:
        raise ValueError("ParaView MultiBlock has no vtkMultiBlockDataSet")
    for node in dataset.findall("./DataSet"):
        xml_blocks.append({
            "index": node.attrib.get("index"),
            "name": node.attrib.get("name"),
            "file": node.attrib.get("file"),
        })
    expected_xml = [{
        "index": str(index),
        "name": row["name"],
        "file": row["file"],
    } for index, row in enumerate(blocks)]
    if xml_blocks != expected_xml:
        raise ValueError("ParaView MultiBlock datasets differ from its manifest")
    names = [row["name"] for row in blocks]
    if names != list(_publication_topologies_from_block_names(names)):
        raise ValueError("ParaView MultiBlock topologies are not in canonical Cartesian order")
    reopened_blocks = []
    if verify_components:
        for row in blocks:
            component = _component_path(index, row["file"])
            reopened = read_paraview(component)
            if type(reopened) is not ReopenedOutput:
                raise TypeError("ParaView MultiBlock component is not a VTU UnstructuredGrid")
            if reopened.output_identity.token != row["output_identity"]:
                raise ValueError("ParaView MultiBlock component identity mismatch")
            geometry = next(iter(reopened.manifest["datasets"]["geometries"].values()))
            if geometry.get("topology") != row["name"]:
                raise ValueError(
                    "ParaView MultiBlock component topology differs from its block name"
                )
            reopened_blocks.append((row["name"], reopened))
    else:
        reopened_blocks = [(row["name"], None) for row in blocks]  # type: ignore[list-item]
    return ReopenedParaViewMultiBlock(
        "vtm",
        manifest_data,
        tuple((name, block) for name, block in reopened_blocks if block is not None)
        if verify_components else tuple(),
        identity,
        index,
    )


def _publication_topologies_from_block_names(names: list[str]) -> tuple[str, ...]:
    allowed = ("cell",) + _FACE_CENTERING_ORDER
    ordered = tuple(name for name in allowed if name in names)
    if sorted(names) != sorted(ordered) or any(name not in allowed for name in names):
        raise ValueError("ParaView MultiBlock names must be cell and/or rank-appropriate face axes")
    return ordered


def _stage_vtm_index(
    target: Path,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    blocks: tuple[tuple[str, _StagedOutputFile], ...],
) -> _StagedOutputFile:
    payload = {
        "schema_version": 1,
        "snapshot": snapshot.to_data(request),
        "blocks": [
            {
                "name": name,
                "file": _relative_component(target.parent, staged.target),
                "output_identity": staged.output_identity.token,
            }
            for name, staged in blocks
        ],
    }
    encoded, token, identity = _index_attributes("paraview-vtm", payload)
    datasets = "".join(
        '<DataSet index="%d" name="%s" file="%s"/>' % (
            index,
            html.escape(name, quote=True),
            html.escape(_relative_component(target.parent, staged.target), quote=True),
        )
        for index, (name, staged) in enumerate(blocks)
    )
    document = f'''<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt32" pops_manifest="{html.escape(encoded, quote=True)}" pops_identity="{html.escape(token, quote=True)}">
  <vtkMultiBlockDataSet>{datasets}</vtkMultiBlockDataSet>
</VTKFile>
'''
    return _stage_text_file(
        target,
        document,
        format_name="paraview-vtu",
        output_identity=identity,
        selection_identity=request.publication_identity,
        verify=lambda path: _read_vtm(path, verify_components=False),
    )


def _series_identity(
    snapshot: OutputSnapshot,
    request: OutputRequest,
    *,
    compression: int | None,
) -> Identity:
    request_data = _series_selection_authority(request.publication_data())
    return make_identity("paraview-series", {
        "consumer": request.consumer_id,
        "request_family": request_data,
        "clock_id": snapshot.clock.clock_id,
        "bind_identity": snapshot.provenance.bind_identity.token,
        "compression": compression,
    })


def _stage_pvtu(
    directory: Path,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    rows: tuple[dict[str, Any], ...],
    series: Identity,
) -> _StagedOutputFile:
    if len(rows) != request.size or any(row["rank"] != rank for rank, row in enumerate(rows)):
        raise ValueError("ParaView PVTU rank envelope is incomplete")
    schemas = [row["schema"] for row in rows]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("ParaView PVTU pieces expose different array schemas")
    pieces = [{
        "rank": row["rank"],
        "file": _relative_component(directory, Path(row["target"])),
        "output_identity": row["output_identity"],
    } for row in rows]
    payload = {
        "schema_version": 1,
        "series_identity": series.token,
        "clock": snapshot.clock.to_data(),
        "schema": schemas[0],
        "pieces": pieces,
    }
    encoded, token, identity = _index_attributes("paraview-pvtu", payload)
    name = "%s__s%09d__%s.pvtu" % (
        _safe_name(request.consumer_id), snapshot.clock.macro_step, identity.hexdigest)
    target = directory / name
    point = schemas[0]["points"]
    point_array = '<PDataArray type="%s" NumberOfComponents="%d"/>' % (
        point["type"], point["components"])
    def parallel_array_nodes(rows: Any) -> str:
        result = []
        for row in rows:
            component_attributes = "".join(
                ' ComponentName%d="%s"' % (index, html.escape(value, quote=True))
                for index, value in enumerate(row["component_names"])
            )
            result.append(
                '<PDataArray type="%s" Name="%s" NumberOfComponents="%d"%s/>' % (
                    row["type"], html.escape(row["name"], quote=True), row["components"],
                    component_attributes,
                ))
        return "".join(result)

    def scalar_attribute(rows: Any) -> str:
        scientific = [
            row["name"] for row in rows
            if row["name"] not in _VTK_RESERVED_ARRAY_NAMES
        ]
        return (
            "" if not scientific
            else ' Scalars="%s"' % html.escape(scientific[0], quote=True)
        )

    point_arrays = schemas[0]["point_arrays"]
    cell_arrays = schemas[0]["cell_arrays"]
    piece_nodes = "".join(
        '<Piece Source="%s"/>' % html.escape(row["file"], quote=True) for row in pieces)
    manifest_xml = html.escape(encoded, quote=True)
    identity_xml = html.escape(token, quote=True)
    point_scalar_xml = scalar_attribute(point_arrays)
    point_arrays_xml = parallel_array_nodes(point_arrays)
    cell_scalar_xml = scalar_attribute(cell_arrays)
    cell_arrays_xml = parallel_array_nodes(cell_arrays)
    document = f'''<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32" pops_manifest="{manifest_xml}" pops_identity="{identity_xml}">
  <PUnstructuredGrid GhostLevel="0"><PPointData{point_scalar_xml}>{point_arrays_xml}</PPointData><PCellData{cell_scalar_xml}>{cell_arrays_xml}</PCellData><PPoints>{point_array}</PPoints>{piece_nodes}</PUnstructuredGrid>
</VTKFile>
'''
    return _stage_text_file(
        target,
        document,
        format_name="paraview-vtu",
        output_identity=identity,
        selection_identity=request.publication_identity,
        verify=lambda path: _read_pvtu(path, verify_components=False),
    )


def _stage_pvd(
    directory: Path,
    snapshot: OutputSnapshot,
    request: OutputRequest,
    component: _StagedOutputFile,
    series: Identity,
    *,
    preset: dict[str, Any],
    pvsm: bool,
) -> _StagedOutputFile:
    prefix = "%s__series-%s" % (_safe_name(request.consumer_id), series.hexdigest)
    target = directory / ("%s__s%09d.pvd" % (prefix, snapshot.clock.macro_step))
    candidates = sorted(directory.glob(prefix + "__s*.pvd"))
    entries: list[dict[str, Any]] = []
    if candidates:
        previous = _read_pvd(candidates[-1], verify_components=False)
        if previous.manifest.get("series_identity") != series.token:
            raise ValueError("latest ParaView PVD belongs to another series")
        entries = [dict(row) for row in previous.manifest["entries"]]
    current = {
        "macro_step": snapshot.clock.macro_step,
        "time_hex": snapshot.clock.time_hex,
        "file": _relative_component(directory, component.target),
        "output_identity": component.output_identity.token,
    }
    if entries and entries[-1]["macro_step"] >= current["macro_step"]:
        if entries[-1] != current:
            raise ValueError("ParaView PVD refuses a non-monotone or conflicting sample")
    else:
        entries.append(current)
    _pvd_time_values(entries)
    payload = {
        "schema_version": 1,
        "series_identity": series.token,
        "entries": entries,
        "presentation": preset,
        "pvsm_requested": pvsm,
    }
    encoded, token, identity = _index_attributes("paraview-pvd", payload)
    datasets = "".join(
        '<DataSet timestep="%s" group="" part="0" file="%s"/>' % (
            f'{float.fromhex(row["time_hex"]):.17g}',
            html.escape(row["file"], quote=True),
        ) for row in entries
    )
    manifest_attribute = html.escape(encoded, quote=True)
    identity_attribute = html.escape(token, quote=True)
    document = f'''<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" pops_manifest="{manifest_attribute}" pops_identity="{identity_attribute}">
  <Collection>{datasets}</Collection>
</VTKFile>
'''
    return _stage_text_file(
        target,
        document,
        format_name="paraview-vtu",
        output_identity=identity,
        selection_identity=request.publication_identity,
        verify=lambda path: _read_pvd(path, verify_components=False),
    )


_PVSM_SAVE_SCRIPT = r'''
import json
import sys
from paraview.simple import (
    ColorBy, GetActiveViewOrCreate, GetAnimationScene, GetColorTransferFunction, OpenDataFile,
    Render, ResetCamera, SaveState, Show, Threshold,
)
pvd, state, raw = sys.argv[1:4]
config = json.loads(raw)
reader = OpenDataFile(pvd)
reader.UpdatePipeline()
source = reader
if config["threshold_active"]:
    source = Threshold(registrationName="PoPS active cells", Input=reader)
    source.Scalars = ["CELLS", "pops_active"]
    source.LowerThreshold = 0.5
view = GetActiveViewOrCreate("RenderView")
display = Show(source, view)
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.GoToLast()
timesteps = list(getattr(reader, "TimestepValues", ()))
if timesteps:
    scene.AnimationTime = timesteps[-1]
    reader.UpdatePipeline(time=timesteps[-1])
display.SetRepresentationType(config["representation"])
association = "POINTS" if config.get("color_association") == "POINTS" else "CELLS"
color = (association, config["color_by"])
if config["component"] is not None:
    color = color + (config["component"],)
ColorBy(display, color)
lut = GetColorTransferFunction(config["color_by"])
if lut.ApplyPreset(config["color_map"], True) is False:
    raise RuntimeError("unknown ParaView color preset: " + config["color_map"])
display.RescaleTransferFunctionToDataRange(True)
display.SetScalarBarVisibility(view, config["show_scalar_bar"])
ResetCamera(view)
Render(view)
SaveState(state)
'''

_PVSM_LOAD_SCRIPT = r'''
import sys
from paraview.simple import LoadState
LoadState(sys.argv[1], data_directory=sys.argv[2], restrict_to_data_directory=True)
'''


def _require_server_manager_state(root: ET.Element) -> None:
    """Accept the two real SaveState envelopes used by supported ParaView releases."""
    if root.tag == "ServerManagerState":
        return
    children = root.findall("./ServerManagerState")
    if root.tag == "GenericParaViewApplication" and len(children) == 1:
        return
    raise ValueError("pvpython SaveState produced an unexpected state document")


def _resolved_preset_data(schema: dict[str, Any], preset: Any) -> dict[str, Any]:
    """Resolve and validate the presentation against the emitted scientific schema."""

    scientific_cells = [
        row for row in schema["cell_arrays"]
        if row["name"] not in _VTK_RESERVED_ARRAY_NAMES
    ]
    scientific_points = [
        row for row in schema.get("point_arrays", [])
        if row["name"] not in _VTK_RESERVED_ARRAY_NAMES
    ]
    scientific = scientific_cells + scientific_points
    if not scientific:
        raise ValueError(
            "ParaView state requires at least one scientific CellData or PointData array"
        )
    color_name = scientific[0]["name"] if preset.color_by is None else preset.color_by
    cell_matches = [row for row in scientific_cells if row["name"] == color_name]
    point_matches = [row for row in scientific_points if row["name"] == color_name]
    if len(cell_matches) + len(point_matches) != 1:
        raise ValueError("ParaViewPreset.color_by does not name one emitted field")
    matches = cell_matches or point_matches
    if preset.component is not None \
            and preset.component not in matches[0]["component_names"]:
        raise ValueError("ParaViewPreset.component is not declared by the selected field")
    active = [row for row in schema["cell_arrays"] if row["name"] == "pops_active"]
    if (
        len(active) > 1
        or active
        and (
            active[0]["components"] != 1
            or active[0]["type"] != "Float64"
            or active[0]["component_names"]
        )
    ):
        raise ValueError("pops_active sidecar has an invalid ParaView scalar schema")
    return dict(
        preset.to_data(),
        color_by=color_name,
        color_association="POINTS" if point_matches else "CELLS",
        threshold_active=bool(active),
    )


def _stage_pvsm(
    pvd: _StagedOutputFile,
    request: OutputRequest,
    schema: dict[str, Any],
    preset: Any,
    pvpython: str,
) -> _StagedOutputFile:
    config = _resolved_preset_data(schema, preset)
    target = pvd.target.with_suffix(".pvsm")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".pops-pvsm-", dir=str(target.parent)) as work:
        generated = Path(work) / target.name
        try:
            subprocess.run(
                [pvpython, "--no-mpi", "-c", _PVSM_SAVE_SCRIPT,
                 str(pvd.target), str(generated),
                 json_text(config)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "pvpython failed while generating ParaView state: %s"
                % _subprocess_detail(exc)
            ) from exc
        if not generated.is_file():
            raise RuntimeError("pvpython SaveState did not create a .pvsm file")
        payload = generated.read_bytes()
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("pvpython SaveState produced invalid XML") from exc
    _require_server_manager_state(root)
    state_data = {
        "schema_version": 1,
        "pvd_identity": pvd.output_identity.token,
        "presentation": config,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    state_identity = make_identity("paraview-pvsm", state_data)

    def verify(path: Any) -> ReopenedParaViewIndex:
        candidate = Path(path)
        data = candidate.read_bytes()
        xml = ET.fromstring(data)
        _require_server_manager_state(xml)
        if hashlib.sha256(data).hexdigest() != state_data["sha256"]:
            raise ValueError("ParaView state failed exact verification")
        return ReopenedParaViewIndex("pvsm", state_data, (pvd.target,), state_identity)

    staged = _stage_text_file(
        target,
        payload,
        format_name="paraview-vtu",
        output_identity=state_identity,
        selection_identity=request.publication_identity,
        verify=verify,
    )
    try:
        subprocess.run(
            [pvpython, "--no-mpi", "-c", _PVSM_LOAD_SCRIPT,
             str(staged.temporary), str(target.parent)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        staged.discard()
        raise RuntimeError(
            "pvpython could not reload the generated ParaView state: %s"
            % _subprocess_detail(exc)
        ) from exc
    return staged


class _ParaViewWriterSession:
    """One transaction owning VTU leaves and their PVTU/PVD/PVSM companions."""

    def __init__(
        self,
        writer: Any,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        communicator: Any,
    ) -> None:
        self._writer = writer
        self._snapshot = snapshot
        self._request = request
        self._communicator = communicator
        self._authority = writer_session_authority(writer.format, request, target)
        self._identity = Identity.from_token(self._authority["session_identity"])
        self._target = Path(target)
        self._series = _series_identity(
            snapshot, request, compression=writer._compression)
        self._vtu: _StagedOutputFile | None = None
        self._companion_vtus: list[_StagedOutputFile] = []
        self._pvtu: _StagedOutputFile | None = None
        self._pvd: _StagedOutputFile | None = None
        self._portable_script: _StagedOutputFile | None = None
        self._portable_manifest: _StagedOutputFile | None = None
        self._pvsm: _StagedOutputFile | None = None
        self._relayed_vtus: list[_StagedOutputFile] = []
        self._rank_rows: tuple[dict[str, Any], ...] = ()
        self._schema: dict[str, Any] | None = None
        self._staged = self._aborted = self._published = self._finalized = False

    @property
    def authority(self) -> dict[str, Any]:
        return dict(self._authority)

    @property
    def identity(self) -> Identity:
        return self._identity

    @property
    def temporary(self) -> Path | None:
        return None if self._vtu is None else self._vtu.temporary

    @property
    def target(self) -> Path:
        return self._target

    def _files(self) -> tuple[_StagedOutputFile, ...]:
        return tuple(
            value for value in (
                self._vtu, *self._companion_vtus, self._pvtu, self._pvd,
                self._portable_script, self._portable_manifest, self._pvsm,
                *self._relayed_vtus,
            )
            if value is not None
        )

    @property
    def recoveries(self) -> tuple[Any, ...]:
        return tuple(
            recovery for staged in self._files() for recovery in staged.recoveries)

    def cleanup_recoveries(self) -> None:
        for staged in self._files():
            staged.cleanup_recoveries()

    def _gather(self, value: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        if self._communicator is None:
            return (value,)
        rows = tuple(allgather_value(self._communicator, value))
        if len(rows) != self._request.size or any(
                not isinstance(row, dict) or row.get("rank") != rank
                for rank, row in enumerate(rows)):
            raise RuntimeError("ParaView companion collective returned malformed rank data")
        return rows

    @staticmethod
    def _raise_failures(phase: str, rows: tuple[dict[str, Any], ...]) -> None:
        failures = [
            "rank %d: %s" % (row["rank"], row["error"])
            for row in rows if row.get("error") is not None
        ]
        if failures:
            raise RuntimeError("ParaView %s failed: %s" % (phase, "; ".join(failures)))

    def _root_phase(self, phase: str, operation: Any) -> None:
        if self._communicator is None:
            operation()
            return
        error = None
        if self._request.rank == 0:
            try:
                operation()
            except BaseException as exc:
                error = _exception_text(exc)
        rows = self._gather({"rank": self._request.rank, "error": error})
        self._raise_failures(phase, rows)

    @staticmethod
    def _rank_row_parts(row: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        """Normalize one rank's relayed files: primary VTU/VTM plus sibling VTU leaves."""

        parts = row.get("parts")
        if parts is None:
            target = Path(row["target"])
            return ({
                "name": target.name,
                "suffix": target.suffix,
                "target": row["target"],
                "output_identity": row["output_identity"],
                "byte_size": int(row["byte_size"]),
            },)
        if not isinstance(parts, list) or not parts:
            raise ValueError("ParaView MPI relay rank has no MultiBlock parts")
        normalized = []
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("ParaView MPI relay part must be a mapping")
            name = Path(str(part.get("name", ""))).name
            suffix = str(part.get("suffix") or Path(name).suffix)
            normalized.append({
                "name": name,
                "suffix": suffix,
                "target": str(part.get("target", name)),
                "output_identity": part["output_identity"],
                "byte_size": int(part["byte_size"]),
            })
        return tuple(normalized)

    def _local_artifact(
        self,
        staged: _StagedOutputFile,
        companions: list[_StagedOutputFile],
    ) -> dict[str, Any]:
        files = (staged, *companions)
        parts = [{
            "name": item.target.name,
            "suffix": item.target.suffix,
            "target": str(item.target.expanduser().resolve()),
            "output_identity": item.output_identity.token,
            "byte_size": item.temporary.stat().st_size,
        } for item in files]
        return {
            "rank": self._request.rank,
            "target": str(staged.target.expanduser().resolve()),
            "output_identity": staged.output_identity.token,
            "schema": self._schema,
            "selection_identity": self._request.publication_identity.token,
            "byte_size": staged.temporary.stat().st_size,
            "parts": parts,
        }

    def _local_relay_files(self, parts: tuple[dict[str, Any], ...]) -> tuple[_StagedOutputFile, ...]:
        if self._vtu is None:
            raise RuntimeError(
                "ParaView MPI relay requires a staged local VTU or MultiBlock on every rank")
        available = {item.target.name: item for item in (self._vtu, *self._companion_vtus)}
        ordered = []
        for part in parts:
            item = available.get(part["name"])
            if item is None:
                raise RuntimeError(
                    "ParaView MPI relay is missing staged local leaf %s" % part["name"])
            ordered.append(item)
        return tuple(ordered)

    def _relay_per_rank_vtus(self) -> None:
        """Copy rank-local VTU or MultiBlock (VTM + sibling VTU leaves) to rank zero."""

        if self._communicator is None or self._vtu is None:
            raise RuntimeError(
                "ParaView MPI relay requires a staged local VTU or MultiBlock on every rank")
        chunk_bytes = self._writer._placement.chunk_bytes
        parts_by_rank = tuple(self._rank_row_parts(row) for row in self._rank_rows)
        if len({len(parts) for parts in parts_by_rank}) != 1:
            raise ValueError("ParaView MPI relay ranks disagree on MultiBlock leaf count")
        local_parts = parts_by_rank[self._request.rank]
        authorities: dict[tuple[int, int], Any] = {}
        streams: dict[tuple[int, int], Any] = {}
        root_targets: dict[tuple[int, int], Path] = {}
        root_error = None
        if self._request.rank == 0:
            try:
                names = []
                for parts in parts_by_rank:
                    for part in parts:
                        name = part["name"]
                        if name in {"", ".", ".."} or part["suffix"] not in {".vtu", ".vtm"}:
                            raise ValueError(
                                "ParaView MPI relay rank target has an invalid basename")
                        names.append(name)
                if len(names) != len(set(names)):
                    raise ValueError("PVTU relay rank targets have colliding basenames")
                for row, parts in zip(self._rank_rows[1:], parts_by_rank[1:], strict=True):
                    owner = int(row["rank"])
                    for part_index, part in enumerate(parts):
                        target = self._target.parent / part["name"]
                        authority = temporary_path(target)
                        key = (owner, part_index)
                        authorities[key] = authority
                        root_targets[key] = target
                        streams[key] = os.fdopen(authority.duplicate(), "wb")
            except BaseException as error:
                root_error = _exception_text(error)
        local_error = None
        local_streams: list[Any] = []
        try:
            try:
                local_files = self._local_relay_files(local_parts)
                local_streams = [item.temporary.open("rb") for item in local_files]
            except BaseException as error:
                local_error = _exception_text(error)
            preparation_rows = self._gather({
                "rank": self._request.rank,
                "error": (root_error or local_error)
                if self._request.rank == 0 else local_error,
            })
            relay_ready = not any(
                row.get("error") is not None for row in preparation_rows)
            if relay_ready:
                if len(local_streams) != len(local_parts):
                    raise RuntimeError("ParaView MPI relay opened a different local leaf count")
                for part_index, local_stream in enumerate(local_streams):
                    maximum_size = max(int(parts[part_index]["byte_size"]) for parts in parts_by_rank)
                    for offset in range(0, maximum_size, chunk_bytes):
                        expected_local = max(0, min(
                            chunk_bytes,
                            int(local_parts[part_index]["byte_size"]) - offset,
                        ))
                        try:
                            payload = local_stream.read(expected_local)
                            if len(payload) != expected_local:
                                raise OSError(
                                    "rank-local VTU ended before its declared byte size")
                        except BaseException as error:
                            if local_error is None:
                                local_error = _exception_text(error)
                            payload = b""
                        gathered = gather_bytes(self._communicator, payload, root=0)
                        if self._request.rank == 0 and root_error is None:
                            if gathered is None:
                                root_error = (
                                    "RuntimeError: PVTU relay root received no byte payloads")
                                continue
                            try:
                                for owner in range(1, self._request.size):
                                    expected = max(0, min(
                                        chunk_bytes,
                                        int(parts_by_rank[owner][part_index]["byte_size"])
                                        - offset,
                                    ))
                                    if len(gathered[owner]) != expected:
                                        raise ValueError(
                                            "PVTU relay rank %d chunk has %d bytes, expected %d"
                                            % (owner, len(gathered[owner]), expected))
                                    streams[(owner, part_index)].write(gathered[owner])
                            except BaseException as error:
                                root_error = _exception_text(error)
        except BaseException as error:
            local_error = local_error or _exception_text(error)
        finally:
            for local_stream in local_streams:
                try:
                    local_stream.close()
                except BaseException as error:
                    local_error = local_error or _exception_text(error)
            if self._request.rank == 0:
                for stream in streams.values():
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                    except BaseException as error:
                        root_error = root_error or _exception_text(error)
        rows = self._gather({
            "rank": self._request.rank,
            "error": (root_error or local_error) if self._request.rank == 0 else local_error,
        })
        try:
            self._raise_failures("VTU MPI relay", rows)
            verification_error = None
            if self._request.rank == 0:
                try:
                    relayed = []
                    rewritten = [dict(self._rank_rows[0])]
                    for row, parts in zip(self._rank_rows[1:], parts_by_rank[1:], strict=True):
                        owner = int(row["rank"])
                        rank_request = replace(self._request, rank=owner)
                        rewritten_parts = []
                        primary_target = None
                        for part_index, part in enumerate(parts):
                            key = (owner, part_index)
                            authority = authorities[key]
                            target = root_targets[key]
                            identity = Identity.from_token(part["output_identity"])
                            if part["suffix"] == ".vtm":
                                reopened = _read_vtm(authority.path, verify_components=False)
                                reopened.require_selection(rank_request)
                                if reopened.output_identity != identity:
                                    raise ValueError(
                                        "relayed VTM output identity differs from rank evidence")
                                verify = lambda path: _read_vtm(path, verify_components=False)
                            else:
                                reopened = read_paraview(authority.path)
                                if type(reopened) is not ReopenedOutput:
                                    raise TypeError(
                                        "relayed ParaView VTU reopened as a collection index")
                                if part_index == 0:
                                    reopened.require_selection(rank_request)
                                if reopened.output_identity != identity:
                                    raise ValueError(
                                        "relayed VTU output identity differs from rank evidence")
                                verify = read_paraview
                            relayed.append(_StagedOutputFile(
                                authority,
                                target,
                                format=self._writer.format,
                                output_identity=identity,
                                selection_identity=rank_request.publication_identity,
                                verify=verify,
                            ))
                            rewritten_parts.append(dict(part, target=str(target.resolve())))
                            if part_index == 0:
                                primary_target = target
                        if primary_target is None:
                            raise RuntimeError("ParaView MPI relay rank has no primary leaf")
                        rewritten.append(dict(
                            row,
                            target=str(primary_target.resolve()),
                            parts=rewritten_parts,
                        ))
                    self._relayed_vtus = relayed
                    self._rank_rows = tuple(rewritten)
                except BaseException as error:
                    verification_error = _exception_text(error)
            verification_rows = self._gather({
                "rank": self._request.rank,
                "error": verification_error,
            })
            self._raise_failures("relayed VTU verification", verification_rows)
        except BaseException as error:
            for authority in authorities.values():
                if authority.is_open:
                    try:
                        _cleanup_staging_authority(
                            authority,
                            replaced_message=(
                                "PVTU relay cleanup refused a replaced temporary at %s"
                                % authority.path),
                        )
                    except BaseException as cleanup_error:
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note("PVTU relay cleanup also failed: %s" % cleanup_error)
            raise

    def stage(self) -> None:
        from pops.output._consumer_contracts import ParallelMode

        if self._aborted or self._finalized:
            raise RuntimeError("aborted ParaView writer session cannot be staged")
        if self._staged:
            return
        active = self._request.parallel_mode is not ParallelMode.ROOT \
            or self._request.rank == 0
        if self._communicator is None:
            if active:
                companions: list[_StagedOutputFile] = []
                staged_vtu: _StagedOutputFile = self._writer._stage_file(
                    self._snapshot, self._request, self._target, companions=companions)
                self._vtu = staged_vtu
                self._companion_vtus = companions
                self._schema = _publication_schema(
                    staged_vtu.temporary,
                    companion_paths=tuple(item.temporary for item in companions),
                    block_names=tuple(
                        item.target.stem.rsplit("__", 1)[-1] for item in companions
                    ),
                )
            rows = ({
                "rank": self._request.rank,
                "artifact": None if self._vtu is None else self._local_artifact(
                    self._vtu, self._companion_vtus),
            },)
        else:
            error = None
            artifact = None
            try:
                if active:
                    companions = []
                    staged_vtu = self._writer._stage_file(
                        self._snapshot, self._request, self._target, companions=companions)
                    self._vtu = staged_vtu
                    self._companion_vtus = companions
                    self._schema = _publication_schema(
                        staged_vtu.temporary,
                        companion_paths=tuple(item.temporary for item in companions),
                        block_names=tuple(
                            item.target.stem.rsplit("__", 1)[-1] for item in companions
                        ),
                    )
                    artifact = self._local_artifact(staged_vtu, companions)
            except BaseException as exc:
                error = _exception_text(exc)
            rows = self._gather({
                "rank": self._request.rank,
                "error": error,
                "artifact": artifact,
            })
            self._raise_failures("VTU staging", rows)
        self._rank_rows = tuple(
            row["artifact"] for row in rows if row["artifact"] is not None)
        if self._request.parallel_mode is ParallelMode.PER_RANK:
            from pops.output.formats import MpiRelayToRoot

            mixed_vtm = any(
                Path(row["target"]).suffix == ".vtm" for row in self._rank_rows
            ) or (self._vtu is not None and self._vtu.target.suffix == ".vtm")
            if type(self._writer._placement) is MpiRelayToRoot:
                self._relay_per_rank_vtus()
            if not mixed_vtm:
                def stage_pvtu() -> None:
                    self._pvtu = _stage_pvtu(
                        self._target.parent,
                        self._snapshot,
                        self._request,
                        self._rank_rows,
                        self._series,
                    )
                self._root_phase("PVTU staging", stage_pvtu)
        if self._writer._collection:
            def stage_pvd() -> None:
                component = self._pvtu if self._pvtu is not None else self._vtu
                if component is None:
                    raise RuntimeError("rank zero has no ParaView component for its collection")
                self._pvd = _stage_pvd(
                    self._target.parent,
                    self._snapshot,
                    self._request,
                    component,
                    self._series,
                    preset=self._writer._preset.to_data(),
                    pvsm=self._writer._pvsm,
                )
                if self._writer._state is not None:
                    schema = self._schema
                    if schema is None:
                        raise RuntimeError(
                            "ParaView state staging requires an authenticated VTU schema")
                    self._portable_script, self._portable_manifest = _stage_portable_state(
                        self._pvd,
                        self._request,
                        schema,
                        self._writer._preset,
                    )
            self._root_phase("PVD staging", stage_pvd)
        self._staged = True

    def abort_prepare(self) -> None:
        if self._aborted:
            return
        failures = []
        for staged in reversed(self._files()):
            try:
                staged.discard()
            except BaseException as exc:
                failures.append(_exception_text(exc))
        self._aborted = True
        if failures:
            raise RuntimeError("ParaView staging cleanup failed: " + "; ".join(failures))

    def publish(self) -> OutputPublicationReceipt | None:
        if not self._staged or self._aborted:
            raise RuntimeError("ParaView writer session must be staged before publication")
        if self._published:
            primary = self._pvsm or self._pvd or self._pvtu or self._vtu
            return None if primary is None else primary.publish()
        if self._communicator is None:
            for staged in self._companion_vtus:
                staged.publish()
            if self._vtu is not None:
                self._vtu.publish()
        else:
            local_error = None
            try:
                for staged in self._companion_vtus:
                    staged.publish()
                if self._vtu is not None:
                    self._vtu.publish()
            except BaseException as exc:
                local_error = _exception_text(exc)
            rows = self._gather({"rank": self._request.rank, "error": local_error})
            self._raise_failures("VTU publication", rows)
        if self._relayed_vtus or (
                self._request.parallel_mode.value == "per_rank"
                and self._writer._placement.to_data()["mode"] == "mpi_relay_to_root"):
            def publish_relayed() -> None:
                for staged in self._relayed_vtus:
                    staged.publish()

            self._root_phase("relayed VTU publication", publish_relayed)
        if self._pvtu is not None or self._request.parallel_mode.value == "per_rank":
            self._root_phase(
                "PVTU publication",
                lambda: self._pvtu.publish() if self._pvtu is not None else None,
            )
        if self._pvd is not None or self._writer._collection:
            self._root_phase(
                "PVD publication",
                lambda: self._pvd.publish() if self._pvd is not None else None,
            )
        if self._writer._state is not None:
            def publish_portable_state() -> None:
                if self._portable_script is None or self._portable_manifest is None:
                    raise RuntimeError("portable ParaView state was not staged on rank zero")
                self._portable_script.publish()
                self._portable_manifest.publish()
                from pops.output.paraview_state import read_portable_paraview_state

                bundle = read_portable_paraview_state(self._portable_manifest.target)
                if bundle.identity != self._portable_manifest.output_identity:
                    raise ValueError(
                        "published portable ParaView state changed scientific identity")

            self._root_phase("portable state publication", publish_portable_state)
        if self._writer._pvsm:
            def publish_pvsm() -> None:
                if self._pvd is None or self._schema is None or self._writer._pvpython is None:
                    raise RuntimeError("ParaView state generation lacks its PVD/schema capability")
                self._pvsm = _stage_pvsm(
                    self._pvd,
                    self._request,
                    self._schema,
                    self._writer._preset,
                    self._writer._pvpython,
                )
                self._pvsm.publish()
            self._root_phase("PVSM publication", publish_pvsm)
        self._published = True
        primary = self._pvsm or self._pvd or self._pvtu or self._vtu
        return None if primary is None else primary.publish()

    def rollback(self) -> None:
        if self._finalized:
            raise RuntimeError("finalized ParaView writer session cannot be rolled back")
        failures = []
        for staged in reversed(self._files()):
            try:
                staged.rollback()
            except BaseException as exc:
                failures.append(_exception_text(exc))
        self._published = False
        self._aborted = True
        if failures:
            raise RuntimeError("ParaView rollback failed: " + "; ".join(failures))

    def finalize(self) -> None:
        if self._finalized:
            return None
        if not self._published:
            raise RuntimeError("only a published ParaView session can be finalized")
        failures = []
        for staged in self._files():
            try:
                staged.finalize()
            except BaseException as exc:
                failures.append(_exception_text(exc))
        self._finalized = True
        if failures:
            raise RuntimeError("ParaView finalization failed: " + "; ".join(failures))
        return None


class ParaViewWriter:
    """Transactional VTU/PVTU/PVD plus portable or materialized ParaView state."""

    format = "paraview-vtu"
    extension = ".vtu"

    def __init__(
        self,
        mode: Any = None,
        *,
        compression: int | None = 6,
        collection: bool = True,
        preset: Any = None,
        placement: Any = None,
        state: Any = None,
    ) -> None:
        from pops.output._consumer_contracts import ParallelMode
        from pops.output.formats import ParaViewPreset, SharedDirectory
        from pops.output.paraview_state import MaterializedPVSM, PortableState

        if mode is None:
            mode = ParallelMode.SERIAL
        if type(mode) is not ParallelMode:
            raise TypeError("ParaViewWriter mode must be an exact ParallelMode")
        if compression is not None and (
                isinstance(compression, bool) or type(compression) is not int
                or compression not in range(10)):
            raise ValueError("ParaViewWriter compression must be None or an integer from 0 to 9")
        if type(collection) is not bool:
            raise TypeError("ParaViewWriter collection must be an exact bool")
        if type(state) not in {type(None), PortableState, MaterializedPVSM}:
            raise TypeError(
                "ParaViewWriter state must be PortableState, MaterializedPVSM, or None")
        if state is not None and not collection:
            raise ValueError("ParaViewWriter state requires a temporal collection")
        resolved_preset = ParaViewPreset() if preset is None else preset
        if type(resolved_preset) is not ParaViewPreset:
            raise TypeError("ParaViewWriter preset must be an exact ParaViewPreset")
        resolved_placement = SharedDirectory() if placement is None else placement
        from pops.output.formats import MpiRelayToRoot

        if type(resolved_placement) not in {SharedDirectory, MpiRelayToRoot}:
            raise TypeError("ParaViewWriter placement has an unsupported type")
        self._mode = mode
        self._compression = compression
        self._collection = collection
        self._preset = resolved_preset
        self._placement = resolved_placement
        self._state = state
        self._pvsm = type(state) is MaterializedPVSM
        self._pvpython_request = (
            state.pvpython if type(state) is MaterializedPVSM else None)
        self._pvpython: str | None = None
        self._pvpython_version: str | None = None
        if self._pvsm:
            self._pvpython, self._pvpython_version = _pvpython_capability(
                self._pvpython_request)

    def preflight(self, execution_context: Any) -> dict[str, Any]:
        from pops.output._consumer_contracts import ParallelMode

        if type(self._mode) is not ParallelMode:
            raise RuntimeError("ParaViewWriter preflight requires its resolved format mode")
        capability = writer_execution_capability(
            execution_context,
            self._mode,
            provider_id="pops.output.paraview-vtu.v1",
        )
        capability["compression"] = self._compression
        capability["collection"] = self._collection
        capability["placement"] = self._placement.to_data()
        capability["state"] = None if self._state is None else self._state.to_data()
        capability["pvsm"] = (
            None if not self._pvsm else {
                "generator": "pvpython.SaveState",
                "version": self._pvpython_version,
            }
        )
        return capability

    def prepare_session(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        communicator: Any = None,
    ) -> _ParaViewWriterSession:
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
        detached_root = request.parallel_mode is ParallelMode.ROOT and request.rank == 0
        if request.parallel_mode is not ParallelMode.SERIAL and communicator is None \
                and not detached_root:
            raise ValueError(
                "distributed ParaView writer session requires its communicator unless a "
                "complete ROOT snapshot was detached for post-commit writing")
        return _ParaViewWriterSession(
            self, snapshot, request, target, communicator)

    def _stage_file(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        companions: list[_StagedOutputFile] | None = None,
    ) -> _StagedOutputFile:
        from pops.output._consumer_contracts import ParallelMode

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("ParaView target must end in .vtu")
        fields = snapshot.select(request)
        _field_families(fields)
        for field in fields:
            complete = request.parallel_mode is not ParallelMode.PER_RANK
            if complete and not field.pieces and field.centering in _FACE_CENTERINGS:
                complete = False
            validate_field_pieces(
                field,
                snapshot.geometry(field.key),
                complete=complete,
                rank=(None if request.parallel_mode is ParallelMode.ROOT else request.rank),
                size=request.size,
            )
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
        geometry_keys = {geometry.key for geometry in geometries}
        sidecars = {
            sidecar.key: sidecar
            for sidecar in snapshot.embedded_boundaries
            if sidecar.key in geometry_keys
        }
        if sidecars and set(sidecars) != geometry_keys:
            raise ValueError(
                "one ParaView artifact cannot mix embedded and non-embedded geometries"
            )
        dimensions = {geometry.spatial_rank for geometry in geometries}
        if len(dimensions) != 1 or not dimensions.issubset({1, 2, 3}):
            raise ValueError(
                "one ParaView VTU snapshot requires one common spatial rank 1, 2, or 3")
        dimension = next(iter(dimensions))
        topologies = _publication_topologies(
            fields, embedded=bool(sidecars), spatial_rank=dimension,
        )
        if any(
            topology != "cell"
            and geometry.coordinate_system == POLAR_ANNULUS_2D_COORDINATES
            and geometry.spatial_rank != 2
            for topology in topologies
            for geometry in geometries
        ):
            raise ValueError("polar-annulus coordinates require spatial rank two")
        if len(topologies) == 1:
            topology = topologies[0]
            return self._stage_unstructured_vtu(
                snapshot,
                request,
                target,
                fields=_fields_for_topology(fields, topology),
                geometries=geometries,
                sidecars=sidecars if topology == "cell" else {},
                topology=topology,
            )
        return self._stage_multiblock(
            snapshot,
            request,
            target,
            fields=fields,
            geometries=geometries,
            sidecars=sidecars,
            topologies=topologies,
            companions=companions,
        )

    def _stage_multiblock(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Path,
        *,
        fields: Any,
        geometries: Any,
        sidecars: dict[Any, Any],
        topologies: tuple[str, ...],
        companions: list[_StagedOutputFile] | None,
    ) -> _StagedOutputFile:
        staged_blocks: list[tuple[str, _StagedOutputFile]] = []
        try:
            for topology in topologies:
                block_fields = _fields_for_topology(fields, topology)
                block_sidecars = sidecars if topology == "cell" else {}
                block_target = target.with_name("%s__%s.vtu" % (target.stem, topology))
                subset = _topology_request(request, block_fields)
                staged = self._stage_unstructured_vtu(
                    snapshot,
                    subset if subset is not None else request,
                    block_target,
                    fields=block_fields,
                    geometries=geometries,
                    sidecars=block_sidecars,
                    topology=topology,
                    verify_selection=subset is not None,
                )
                staged_blocks.append((topology, staged))
            vtm = _stage_vtm_index(
                target.with_suffix(".vtm"),
                snapshot,
                request,
                tuple(staged_blocks),
            )
        except BaseException:
            for _name, staged in staged_blocks:
                try:
                    staged.discard()
                except BaseException:
                    pass
            raise
        if companions is not None:
            companions.extend(staged for _name, staged in staged_blocks)
        return vtm

    def _stage_unstructured_vtu(
        self,
        snapshot: OutputSnapshot,
        request: OutputRequest,
        target: Any,
        *,
        fields: Any,
        geometries: Any,
        sidecars: dict[Any, Any],
        topology: str,
        verify_selection: bool = True,
    ) -> _StagedOutputFile:
        from pops.output._consumer_contracts import ParallelMode

        import numpy as np

        target = Path(target)
        if target.suffix != self.extension:
            raise ValueError("ParaView target must end in .vtu")
        families = _field_families(fields)
        face_centering = None if topology == "cell" else topology
        if sidecars and face_centering is not None:
            raise NotImplementedError(
                "ParaView VTU will not attach cell-centered embedded-boundary sidecars to a "
                "face mesh; that would silently recenter face locations onto primal cells"
            )
        dimension = geometries[0].spatial_rank
        face_normal = (
            0 if face_centering is None
            else _face_array_axis(dimension, face_centering)
        )
        if face_centering is None:
            vtk_cell_type, points_per_cell = _VTK_CELL[dimension]
            corner_offsets = _cell_corner_offsets(dimension)
        else:
            vtk_cell_type, points_per_cell = _VTK_FACE[dimension]
            corner_offsets = _face_corner_offsets(dimension, face_normal)
        emission_masks = {}
        replication_masks = {}
        for geometry in geometries:
            emission_shape = (
                geometry.cell_shape if face_centering is None
                else _centering_shape(geometry.cell_shape, face_centering)
            )
            if request.parallel_mode is not ParallelMode.PER_RANK:
                emission_masks[geometry.key] = (
                    geometry.valid_cells if face_centering is None
                    else _staggered_face_mask(geometry.valid_cells, face_normal)
                )
                replication_masks[geometry.key] = np.zeros(
                    emission_shape, dtype=np.bool_)
                continue
            masks = []
            replicated_masks = []
            for field in fields:
                if snapshot.geometry(field.key).key != geometry.key:
                    continue
                if face_centering is not None and field.centering == "node":
                    continue
                mask = np.zeros(emission_shape, dtype=np.bool_)
                replicated = np.zeros(emission_shape, dtype=np.bool_)
                for piece in field.pieces:
                    if field.centering == "cell" or field.centering in _FACE_CENTERINGS:
                        spatial = _spatial_slices(piece.lower, piece.upper)
                    else:
                        box = geometry.boxes[piece.global_box_index]
                        spatial = _spatial_slices(
                            box[:dimension], box[dimension:])
                    mask[spatial] = True
                    if piece.replicated:
                        replicated[spatial] = True
                masks.append(mask)
                replicated_masks.append(replicated)
            if masks and any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
                raise ValueError(
                    "PER_RANK ParaView fields disagree on their local geometry ownership")
            if replicated_masks and any(
                    not np.array_equal(mask, replicated_masks[0])
                    for mask in replicated_masks[1:]):
                raise ValueError(
                    "PER_RANK ParaView fields disagree on replicated local geometry")
            emission_masks[geometry.key] = (
                masks[0] if masks else np.zeros(emission_shape, dtype=np.bool_))
            replication_masks[geometry.key] = (
                replicated_masks[0]
                if replicated_masks else np.zeros(emission_shape, dtype=np.bool_)
            )
        cell_counts = [
            int(np.count_nonzero(emission_masks[geometry.key]))
            for geometry in geometries
        ]
        point_masks = {}
        point_counts = []
        for geometry in geometries:
            used = (
                _point_mask(emission_masks[geometry.key])
                if face_centering is None
                else _face_point_mask(emission_masks[geometry.key], face_normal)
            )
            point_masks[geometry.key] = used
            point_counts.append(int(np.count_nonzero(used)))
        n_cells = sum(cell_counts)
        n_points = sum(point_counts)
        points = np.empty((n_points, 3), dtype="<f8")
        connectivity = np.empty(n_cells * points_per_cell, dtype="<i8")
        layout_ordinals = np.empty(n_cells, dtype="<i4")
        levels = np.empty(n_cells, dtype="<i4")
        covered = np.empty(n_cells, dtype="u1")
        ghost_types = np.empty(n_cells, dtype="u1")
        volumes = np.empty(n_cells, dtype="<f8")
        geometry_ranges = {}
        cell = 0
        point = 0
        for ordinal, (geometry, count, point_count) in enumerate(
            zip(geometries, cell_counts, point_counts, strict=True)
        ):
            start = cell
            cell = start + count
            cell_indices = np.nonzero(emission_masks[geometry.key])
            used = point_masks[geometry.key]
            logical_points = np.nonzero(used)
            point_end = point + point_count
            points[point:point_end, :] = _physical_point_coordinates(
                geometry, logical_points)
            point_ids = np.full(used.shape, -1, dtype="<i8")
            point_ids[used] = np.arange(point, point_end, dtype="<i8")
            cell_connectivity = connectivity[
                start * points_per_cell:cell * points_per_cell
            ].reshape((count, points_per_cell))
            for corner_index, corner in enumerate(corner_offsets):
                vertex = tuple(
                    indices + offset
                    for indices, offset in zip(cell_indices, corner, strict=True))
                cell_connectivity[:, corner_index] = point_ids[vertex]
            point = point_end
            layout_ordinals[start:cell] = ordinal
            levels[start:cell] = geometry.level
            if face_centering is None:
                covered[start:cell] = geometry.coverage[cell_indices]
                ghost_types[start:cell] = covered[start:cell] * _VTK_REFINED_CELL
                volumes[start:cell] = geometry.cell_volumes[cell_indices]
            else:
                covered[start:cell] = 0
                ghost_types[start:cell] = 0
                volumes[start:cell] = 0.0
            if request.parallel_mode is ParallelMode.PER_RANK and request.rank != 0:
                ghost_types[start:cell] |= (
                    replication_masks[geometry.key][cell_indices].astype("u1")
                    * np.uint8(_VTK_DUPLICATE_CELL)
                )
            geometry_ranges[geometry.key] = {
                "cell": (start, cell),
                "point": (point_end - point_count, point_end),
                "ordinal": ordinal,
            }
        arrays = {
            "Points": points,
            "connectivity": connectivity,
            "offsets": np.arange(
                points_per_cell,
                n_cells * points_per_cell + 1,
                points_per_cell,
                dtype="<i8",
            ),
            "types": np.full(n_cells, vtk_cell_type, dtype="u1"),
            "pops_layout": layout_ordinals,
            "pops_level": levels,
            "vtkGhostType": ghost_types,
            "TimeValue": np.asarray(
                [float.fromhex(snapshot.clock.time_hex)], dtype="<f8"),
        }
        if face_centering is None:
            arrays["pops_coverage"] = covered
            arrays["pops_cell_volume"] = volumes
        datasets = {"fields": {}, "geometries": {}, "embedded_boundaries": {}}
        for ordinal, geometry in enumerate(geometries):
            ranges = geometry_ranges[geometry.key]
            datasets["geometries"]["%s#%d" % geometry.key] = {
                "layout_ordinal": ordinal,
                "spatial_rank": dimension,
                "topology": "cell" if face_centering is None else face_centering,
                "cell_range": list(ranges["cell"]),
                "point_range": list(ranges["point"]),
            }
        display_names = _field_display_names(families)
        field_components = {}
        cell_field_names = []
        point_field_names = []
        for name, (_family, members) in zip(display_names, families, strict=True):
            first = members[0]
            field_components[name] = first.component_names
            components = len(first.component_names)
            association = "point" if first.centering == "node" else "cell"
            tuple_count = n_cells if association == "cell" else n_points
            shape = (tuple_count, components) if components else (tuple_count,)
            combined = np.empty(shape, dtype=np.dtype(first.array_dtype))
            written = np.zeros(tuple_count, dtype=np.bool_)
            for field in members:
                geometry = snapshot.geometry(field.key)
                valid = (
                    emission_masks[geometry.key]
                    if association == "cell" else point_masks[geometry.key]
                )
                ranges = geometry_ranges[geometry.key]
                start, end = ranges[association]
                ordinal = ranges["ordinal"]
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
                    "display_name": name,
                    "association": association,
                    "layout_ordinal": ordinal,
                    association + "_range": [start, end],
                    "array": array_evidence(values),
                }
            if not np.all(written):
                raise ValueError(
                    "ParaView logical field family does not cover every emitted geometry")
            combined.setflags(write=False)
            arrays[name] = combined
            if association == "cell":
                cell_field_names.append(name)
            else:
                point_field_names.append(name)
        if sidecars:
            for geometry in geometries:
                ranges = geometry_ranges[geometry.key]
                datasets["embedded_boundaries"]["%s#%d" % geometry.key] = {
                    "layout_ordinal": ranges["ordinal"],
                    "cell_range": list(ranges["cell"]),
                    "arrays": {},
                }
            for name in EMBEDDED_BOUNDARY_ARRAY_NAMES:
                combined = np.empty(n_cells, dtype=np.float64)
                for geometry in geometries:
                    ranges = geometry_ranges[geometry.key]
                    start, end = ranges["cell"]
                    values = _embedded_values_on_mask(
                        sidecars[geometry.key],
                        name,
                        emission_masks[geometry.key],
                        require_piece_subset=True,
                    )
                    combined[start:end] = values
                    datasets["embedded_boundaries"]["%s#%d" % geometry.key]["arrays"][name] = (
                        array_evidence(values)
                    )
                combined.setflags(write=False)
                arrays[name] = combined
        evidence = {name: array_evidence(value) for name, value in arrays.items()}
        output_manifest, identity = manifest(
            self.format, snapshot, request, evidence, datasets=datasets)
        encoded_arrays = {
            name: _vtk_array(value, self._compression) for name, value in arrays.items()
        }
        temporary = temporary_path(target)
        point_type, point_data = encoded_arrays["Points"]
        cell_names = (
            ("pops_layout", "pops_level", "vtkGhostType")
            if face_centering is not None
            else (
                "pops_layout", "pops_level", "pops_coverage", "vtkGhostType",
                "pops_cell_volume",
            )
        ) + (EMBEDDED_BOUNDARY_ARRAY_NAMES if sidecars else ()) + tuple(cell_field_names)

        def data_arrays(names: Any) -> str:
            result = []
            for name in names:
                vtk_type, encoded = encoded_arrays[name]
                components = arrays[name].shape[1] if arrays[name].ndim == 2 else 1
                component_attributes = "".join(
                    ' ComponentName%d="%s"' % (index, html.escape(component, quote=True))
                    for index, component in enumerate(field_components.get(name, ()))
                )
                result.append(
                    '<DataArray type="%s" Name="%s" NumberOfComponents="%d" '
                    'format="binary"%s>%s</DataArray>'
                    % (
                        vtk_type, html.escape(name, quote=True), components,
                        component_attributes, encoded,
                    ))
            return "".join(result)

        cells = []
        for name in ("connectivity", "offsets", "types"):
            vtk_type, encoded = encoded_arrays[name]
            cells.append(
                '<DataArray type="%s" Name="%s" format="binary">%s</DataArray>'
                % (vtk_type, name, encoded))
        time_type, time_data = encoded_arrays["TimeValue"]
        compressor = (
            '' if self._compression is None
            else ' compressor="vtkZLibDataCompressor"'
        )
        cell_scalar_attribute = (
            '' if not cell_field_names
            else ' Scalars="%s"' % html.escape(cell_field_names[0], quote=True)
        )
        point_scalar_attribute = (
            '' if not point_field_names
            else ' Scalars="%s"' % html.escape(point_field_names[0], quote=True)
        )
        document = '''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32"{compressor}>
  <UnstructuredGrid><FieldData><DataArray type="{time_type}" Name="TimeValue" NumberOfTuples="1" NumberOfComponents="1" format="binary">{time_data}</DataArray></FieldData><Piece NumberOfPoints="{points}" NumberOfCells="{cells_count}">
    <FieldData><DataArray type="String" Name="pops_output_manifest" NumberOfTuples="1" format="ascii">{manifest}</DataArray></FieldData>
    <Points><DataArray type="{point_type}" NumberOfComponents="3" format="binary">{point_data}</DataArray></Points>
    <Cells>{cells}</Cells><PointData{point_scalar_attribute}>{point_data_arrays}</PointData><CellData{cell_scalar_attribute}>{cell_data}</CellData>
  </Piece></UnstructuredGrid>
</VTKFile>
'''.format(
            points=len(points),
            cells_count=n_cells,
            compressor=compressor,
            cell_scalar_attribute=cell_scalar_attribute,
            point_scalar_attribute=point_scalar_attribute,
            time_type=time_type,
            time_data=time_data,
            manifest=html.escape(json_text(output_manifest)),
            point_type=point_type,
            point_data=point_data,
            cells="".join(cells),
            point_data_arrays=data_arrays(point_field_names),
            cell_data=data_arrays(cell_names),
        )
        try:
            with os.fdopen(
                temporary.duplicate(), "w", encoding="utf-8", newline="\n",
            ) as stream:
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            reopened = read_paraview(temporary.path)
            if type(reopened) is not ReopenedOutput:
                raise TypeError("staged ParaView VTU reopened as a collection index")
            if verify_selection:
                reopened.require_selection(request)
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


def read_paraview(path: Any) -> ReopenedOutput | ReopenedParaViewIndex | ReopenedParaViewMultiBlock:
    root = ET.parse(path).getroot()
    if root.tag == "VTKFile" and root.attrib.get("type") == "PUnstructuredGrid":
        return _read_pvtu(path, verify_components=True)
    if root.tag == "VTKFile" and root.attrib.get("type") == "Collection":
        return _read_pvd(path, verify_components=True)
    if root.tag == "VTKFile" and root.attrib.get("type") == "vtkMultiBlockDataSet":
        return _read_vtm(path, verify_components=True)
    if root.tag != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("ParaView output is not a VTK UnstructuredGrid")
    if root.attrib.get("byte_order") != "LittleEndian" \
            or root.attrib.get("header_type") != "UInt32":
        raise ValueError("ParaView output has unsupported binary framing")
    compressor = root.attrib.get("compressor")
    if compressor not in (None, "vtkZLibDataCompressor"):
        raise ValueError("ParaView output has an unsupported compressor")
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
        arrays[name] = _decode_vtk_array(node, evidence, compressor=compressor)
    points = root.find(".//Points/DataArray")
    if points is None:
        raise ValueError("ParaView output has no point geometry")
    arrays["Points"] = _decode_vtk_array(
        points, output_manifest["arrays"]["Points"], compressor=compressor)
    for node in root.findall(".//Cells/DataArray"):
        name = node.attrib["Name"]
        evidence = output_manifest["arrays"].get(name)
        if evidence is None:
            raise ValueError("ParaView array %r is unauthenticated" % name)
        arrays[name] = _decode_vtk_array(node, evidence, compressor=compressor)
    associations = {}
    for association, xpath in (
        ("point", ".//PointData/DataArray"),
        ("cell", ".//CellData/DataArray"),
    ):
        for node in root.findall(xpath):
            name = node.attrib["Name"]
            evidence = output_manifest["arrays"].get(name)
            if evidence is None:
                raise ValueError("ParaView array %r is unauthenticated" % name)
            if name in associations:
                raise ValueError("ParaView data array %r has duplicate associations" % name)
            associations[name] = association
            arrays[name] = _decode_vtk_array(node, evidence, compressor=compressor)
    if set(arrays) != set(output_manifest["arrays"]):
        raise ValueError("ParaView arrays differ from its exact manifest")
    for name, evidence in output_manifest["arrays"].items():
        if array_evidence(arrays[name]) != evidence:
            raise ValueError("ParaView array %r failed verification" % name)
    for record in output_manifest["datasets"]["fields"].values():
        association = record.get("association", "cell")
        if association not in {"cell", "point"}:
            raise ValueError("ParaView selected field has an invalid data association")
        if associations.get(record["name"]) != association:
            raise ValueError("ParaView selected field differs from its exact data association")
        start, end = record[association + "_range"]
        values = arrays[record["name"]][start:end]
        if array_evidence(values) != record["array"]:
            raise ValueError("ParaView selected field failed verification")
    return ReopenedOutput(output_manifest, arrays, identity)


def read_paraview_parallel(path: Any) -> ReopenedParaViewIndex:
    return _read_pvtu(path, verify_components=True)


def _resolve_paraview_series_path(path: Any) -> Path:
    """Resolve one directory to the latest unambiguous authenticated PVD index."""
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("ParaView series path must not be a symbolic link")
    if not unresolved.is_dir():
        if unresolved.parent.is_symlink():
            raise ValueError("ParaView series parent must not be a symbolic link")
        return unresolved

    candidates = tuple(sorted(unresolved.glob("*.pvd"), key=lambda item: item.name))
    if not candidates:
        raise FileNotFoundError("ParaView output directory has no PVD time series")

    inspected = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("ParaView PVD candidate must be one regular file")
        reopened = _read_pvd(candidate, verify_components=False)
        series_token = reopened.manifest.get("series_identity")
        try:
            series_identity = Identity.from_token(series_token)
        except (TypeError, ValueError) as exc:
            raise ValueError("ParaView PVD has no valid series identity") from exc
        if series_identity.domain != "paraview-series":
            raise ValueError("ParaView PVD has an invalid series identity domain")
        inspected.append((candidate, reopened, series_identity))

    series_identities = {identity for _, _, identity in inspected}
    if len(series_identities) != 1:
        raise ValueError(
            "ParaView output directory has multiple PVD time series; "
            "pass one exact .pvd path"
        )

    largest_size = max(len(reopened.manifest["entries"])
                       for _, reopened, _ in inspected)
    largest = [
        (candidate, reopened)
        for candidate, reopened, _ in inspected
        if len(reopened.manifest["entries"]) == largest_size
    ]
    expected_entries = largest[0][1].manifest["entries"]
    if any(reopened.manifest["entries"] != expected_entries
           for _, reopened in largest[1:]):
        raise ValueError(
            "ParaView output directory has divergent PVD histories; "
            "pass one exact .pvd path"
        )
    for _, reopened, _ in inspected:
        entries = reopened.manifest["entries"]
        if entries != expected_entries[:len(entries)]:
            raise ValueError(
                "ParaView output directory has divergent PVD histories; "
                "pass one exact .pvd path"
            )
    return largest[0][0]


def read_paraview_series(path: Any) -> ReopenedParaViewIndex:
    return _read_pvd(_resolve_paraview_series_path(path), verify_components=True)


__all__ = [
    "ParaViewWriter", "ReopenedParaViewIndex", "ReopenedParaViewMultiBlock",
    "read_paraview",
    "read_paraview_parallel", "read_paraview_series",
]
