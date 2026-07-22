"""Versioned, pickle-free archive codec for complete post-commit observer frames."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pops.identity import Identity, make_identity
from pops.model import Handle
from pops.output._consumer_contracts import ParallelMode
from pops.output.data import (
    ArrayPiece,
    DiagnosticKey,
    DiagnosticPayload,
    FieldKey,
    FieldPayload,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
    _NATIVE_GEOMETRY_ARRAYS,
    _NativeCompositeIntegral,
    array_evidence,
)
from pops.output.observers import ObserverFrame


_SCHEMA_VERSION = 1
_FORMAT = "pops-observer-frame-archive"
_MANIFEST_MEMBER = "manifest.json"
_ARRAY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o100600 << 16


def _mapping(value: Any, keys: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise TypeError("%s has an unsupported schema" % where)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("observer archive JSON contains a duplicate key %r" % key)
        result[key] = value
    return result


def _decode_json(payload: bytes) -> dict[str, Any]:
    try:
        result = json.loads(payload.decode("ascii"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("observer archive manifest is not canonical JSON") from error
    if type(result) is not dict or _json_bytes(result) != payload:
        raise ValueError("observer archive manifest is not canonical JSON")
    return result


def _npy_bytes(value: Any) -> bytes:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("observer archive arrays cannot use object dtype")
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _zip_member(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = _ZIP_EXTERNAL_ATTR
    return info, payload


def _field_key(data: Any) -> FieldKey:
    row = _mapping(data, {
        "reference", "component_manifest_identity", "layout_identity", "level", "state_id",
    }, "observer archive FieldKey")
    result = FieldKey(
        Handle.from_canonical_identity(row["reference"]),
        Identity.from_token(row["component_manifest_identity"]),
        Identity.from_token(row["layout_identity"]),
        row["level"],
        row["state_id"],
    )
    if result.to_data() != dict(row):
        raise ValueError("observer archive FieldKey is not canonical")
    return result


def _diagnostic_key(data: Any) -> DiagnosticKey:
    row = _mapping(data, {
        "reference", "component_manifest_identity", "layout_identity", "level", "state_id",
        "reduction",
    }, "observer archive DiagnosticKey")
    result = DiagnosticKey(
        Handle.from_canonical_identity(row["reference"]),
        Identity.from_token(row["component_manifest_identity"]),
        Identity.from_token(row["layout_identity"]),
        row["level"],
        row["state_id"],
        row["reduction"],
    )
    if result.to_data() != dict(row):
        raise ValueError("observer archive DiagnosticKey is not canonical")
    return result


def _clock(data: Any) -> OutputClock:
    row = _mapping(data, {
        "clock_id", "time", "macro_step", "stage", "tick", "level", "substep",
        "stage_index", "fraction", "dt",
    }, "observer archive clock")
    fraction = row["fraction"]
    if not isinstance(fraction, list) or len(fraction) != 2:
        raise TypeError("observer archive clock fraction must be one integer pair")
    result = OutputClock(
        row["clock_id"], row["time"], row["macro_step"], row["stage"], row["tick"],
        row["level"], row["substep"], row["stage_index"], fraction[0], fraction[1], row["dt"],
    )
    if result.to_data() != dict(row):
        raise ValueError("observer archive clock is not canonical")
    return result


def _provenance(data: Any) -> OutputProvenance:
    row = _mapping(data, {
        "plan_identity", "bind_identity", "run_identity", "source",
    }, "observer archive provenance")
    result = OutputProvenance(
        Identity.from_token(row["plan_identity"]),
        Identity.from_token(row["bind_identity"]),
        Identity.from_token(row["run_identity"]),
        row["source"],
    )
    if result.to_data() != dict(row):
        raise ValueError("observer archive provenance is not canonical")
    return result


def _diagnostic(data: Any) -> DiagnosticPayload:
    row = _mapping(data, {"key", "value", "units", "terms"}, "observer archive diagnostic")
    if not isinstance(row["value"], str) or not isinstance(row["terms"], Mapping):
        raise TypeError("observer archive diagnostic values must use float.hex strings")
    try:
        value = float.fromhex(row["value"])
        terms = {
            name: float.fromhex(item)
            for name, item in row["terms"].items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("observer archive diagnostic contains invalid float.hex data") from error
    result = DiagnosticPayload(_diagnostic_key(row["key"]), value, row["units"], terms)
    if result.to_data() != dict(row):
        raise ValueError("observer archive diagnostic is not canonical")
    return result


def _request(data: Any) -> OutputRequest:
    row = _mapping(data, {
        "consumer_id", "selection", "parallel_mode", "rank", "size", "diagnostics",
    }, "observer archive request")
    if not isinstance(row["selection"], list) or not isinstance(row["diagnostics"], list):
        raise TypeError("observer archive request selections must be lists")
    try:
        mode = ParallelMode(row["parallel_mode"])
    except (TypeError, ValueError) as error:
        raise ValueError("observer archive request has an unsupported parallel mode") from error
    result = OutputRequest(
        row["consumer_id"],
        tuple(_field_key(item) for item in row["selection"]),
        mode,
        row["rank"],
        row["size"],
        tuple(_diagnostic_key(item) for item in row["diagnostics"]),
    )
    if result.to_data() != dict(row):
        raise ValueError("observer archive request is not canonical")
    return result


def _frame_projection(
    frame: ObserverFrame,
    add_array: Callable[[str, Any], None],
) -> dict[str, Any]:
    geometries = []
    for index, geometry in enumerate(frame.snapshot.geometries):
        prefix = "geometry_%04d" % index
        arrays = {
            "valid_cells": prefix + "_valid_cells",
            "coverage": prefix + "_coverage",
            "cell_volumes": prefix + "_cell_volumes",
        }
        add_array(arrays["valid_cells"], geometry.valid_cells)
        add_array(arrays["coverage"], geometry.coverage)
        add_array(arrays["cell_volumes"], geometry.cell_volumes)
        geometries.append({
            "layout_identity": geometry.layout_identity.token,
            "layout_kind": geometry.layout_kind,
            "level": geometry.level,
            "origin": [value.hex() for value in geometry.origin],
            "spacing": [value.hex() for value in geometry.spacing],
            "cell_shape": list(geometry.cell_shape),
            "boxes": [list(box) for box in geometry.boxes],
            "coordinate_system": geometry.coordinate_system,
            "cell_measure": geometry.cell_measure,
            "axis_names": list(geometry.axis_names),
            "arrays": arrays,
        })
    fields = []
    for field_index, field in enumerate(frame.snapshot.fields):
        pieces = []
        for piece_index, piece in enumerate(field.pieces):
            array_name = "field_%04d_piece_%04d" % (field_index, piece_index)
            add_array(array_name, piece.values)
            pieces.append({
                "lower": list(piece.lower),
                "upper": list(piece.upper),
                "global_box_index": piece.global_box_index,
                "owner_rank": piece.owner_rank,
                "replicated": piece.replicated,
                "array": array_name,
            })
        fields.append({
            "key": field.key.to_data(),
            "centering": field.centering,
            "units": field.units,
            "component_names": list(field.component_names),
            "global_shape": list(field.global_shape),
            "dtype": field.array_dtype,
            "pieces": pieces,
        })
    native_integrals = [{
        "family_identity": item.family_identity.token,
        "levels": list(item.levels),
        "value": item.value.hex(),
    } for item in frame.snapshot._native_composite_integrals]
    return {
        "clock": frame.snapshot.clock.to_data(),
        "provenance": frame.snapshot.provenance.to_data(),
        "geometries": geometries,
        "fields": fields,
        "diagnostics": [item.to_data() for item in frame.snapshot.diagnostics],
        "native_composite_integrals": native_integrals,
        "metadata": dict(frame.snapshot.metadata),
        "request": frame.request.to_data(),
    }


def observer_archive_identity(frame: ObserverFrame) -> Identity:
    """Return the identity of the complete frame, including unselected arrays and diagnostics."""
    if type(frame) is not ObserverFrame:
        raise TypeError("observer archive requires an exact ObserverFrame")
    evidence: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if name in evidence:
            raise RuntimeError("observer archive array name collision")
        evidence[name] = array_evidence(value)

    projection = _frame_projection(frame, add)
    return make_identity("observer-frame-archive", {
        "schema_version": _SCHEMA_VERSION,
        "frame": projection,
        "arrays": evidence,
    })


def encode_observer_frame(frame: ObserverFrame) -> bytes:
    """Encode one complete detached observer frame as deterministic ZIP/NPY bytes."""
    if type(frame) is not ObserverFrame:
        raise TypeError("encode_observer_frame requires an exact ObserverFrame")
    arrays: dict[str, Any] = {}
    evidence: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if _ARRAY_NAME.fullmatch(name) is None or name in arrays:
            raise RuntimeError("observer archive produced an invalid array name")
        arrays[name] = value
        evidence[name] = array_evidence(value)

    projection = _frame_projection(frame, add)
    identity_payload = {
        "schema_version": _SCHEMA_VERSION,
        "frame": projection,
        "arrays": evidence,
    }
    archive_identity = make_identity("observer-frame-archive", identity_payload)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "format": _FORMAT,
        "frame_identity": frame.identity.token,
        "archive_identity": archive_identity.token,
        "frame": projection,
        "arrays": {
            name: {
                "member": "arrays/%s.npy" % name,
                "evidence": evidence[name],
            }
            for name in sorted(arrays)
        },
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(*_zip_member(_MANIFEST_MEMBER, _json_bytes(manifest)))
        for name in sorted(arrays):
            archive.writestr(*_zip_member(
                manifest["arrays"][name]["member"], _npy_bytes(arrays[name])))
    return output.getvalue()


def _array_members(archive: zipfile.ZipFile, manifest: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    descriptions = manifest["arrays"]
    if not isinstance(descriptions, Mapping):
        raise TypeError("observer archive arrays manifest must be a mapping")
    arrays = {}
    expected_members = {_MANIFEST_MEMBER}
    for name, raw in descriptions.items():
        if not isinstance(name, str) or _ARRAY_NAME.fullmatch(name) is None:
            raise ValueError("observer archive array name is not canonical")
        row = _mapping(raw, {"member", "evidence"}, "observer archive array description")
        member = row["member"]
        if member != "arrays/%s.npy" % name:
            raise ValueError("observer archive array member is not canonical")
        evidence = _mapping(
            row["evidence"], {"dtype", "shape", "content_sha256"},
            "observer archive array evidence")
        expected_members.add(member)
        try:
            payload = archive.read(member)
            loaded = np.load(io.BytesIO(payload), allow_pickle=False)
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
            raise ValueError("observer archive array %r cannot be decoded" % name) from error
        if not isinstance(loaded, np.ndarray) or loaded.dtype.hasobject:
            raise TypeError("observer archive array %r is not a plain ndarray" % name)
        array = np.ascontiguousarray(loaded)
        if _npy_bytes(array) != payload:
            raise ValueError("observer archive array %r is not canonical NPY" % name)
        if array_evidence(array) != dict(evidence):
            raise ValueError("observer archive array %r failed SHA-256 verification" % name)
        array.setflags(write=False)
        arrays[name] = array
    if set(archive.namelist()) != expected_members:
        raise ValueError("observer archive members differ from its exact manifest")
    return arrays


def _decode_geometry(data: Any, arrays: Mapping[str, Any]) -> LevelGeometry:
    row = _mapping(data, {
        "layout_identity", "layout_kind", "level", "origin", "spacing", "cell_shape",
        "boxes", "coordinate_system", "cell_measure", "axis_names", "arrays",
    }, "observer archive geometry")
    names = _mapping(
        row["arrays"], {"valid_cells", "coverage", "cell_volumes"},
        "observer archive geometry arrays")
    try:
        origin = tuple(float.fromhex(value) for value in row["origin"])
        spacing = tuple(float.fromhex(value) for value in row["spacing"])
        geometry = LevelGeometry(
            Identity.from_token(row["layout_identity"]),
            row["layout_kind"],
            row["level"],
            origin,
            spacing,
            tuple(row["cell_shape"]),
            tuple(tuple(box) for box in row["boxes"]),
            arrays[names["coverage"]],
            arrays[names["cell_volumes"]],
            coordinate_system=row["coordinate_system"],
            cell_measure=row["cell_measure"],
            axis_names=tuple(row["axis_names"]),
            _native_valid_cells=arrays[names["valid_cells"]],
            _native_arrays=_NATIVE_GEOMETRY_ARRAYS,
        )
    except KeyError as error:
        raise ValueError("observer archive geometry references an unknown array") from error
    import numpy as np

    if not np.array_equal(geometry.valid_cells, arrays[names["valid_cells"]]):
        raise ValueError("observer archive geometry valid-cell mask differs from its boxes")
    return geometry


def _decode_field(data: Any, arrays: Mapping[str, Any]) -> FieldPayload:
    row = _mapping(data, {
        "key", "centering", "units", "component_names", "global_shape", "dtype", "pieces",
    }, "observer archive field")
    if not isinstance(row["pieces"], list):
        raise TypeError("observer archive field pieces must be a list")
    pieces = []
    for raw in row["pieces"]:
        piece = _mapping(raw, {
            "lower", "upper", "global_box_index", "owner_rank", "replicated", "array",
        }, "observer archive array piece")
        try:
            value = arrays[piece["array"]]
        except KeyError as error:
            raise ValueError("observer archive field references an unknown array") from error
        pieces.append(ArrayPiece(
            tuple(piece["lower"]), tuple(piece["upper"]), value,
            piece["global_box_index"], piece["owner_rank"], piece["replicated"],
        ))
    return FieldPayload(
        _field_key(row["key"]), row["centering"], row["units"],
        tuple(row["component_names"]), tuple(row["global_shape"]), tuple(pieces),
        dtype=row["dtype"],
    )


def _decode_frame(data: Any, arrays: Mapping[str, Any]) -> ObserverFrame:
    row = _mapping(data, {
        "clock", "provenance", "geometries", "fields", "diagnostics",
        "native_composite_integrals", "metadata", "request",
    }, "observer archive frame")
    for name in ("geometries", "fields", "diagnostics", "native_composite_integrals"):
        if not isinstance(row[name], list):
            raise TypeError("observer archive frame %s must be a list" % name)
    native_integrals = []
    for raw in row["native_composite_integrals"]:
        item = _mapping(
            raw, {"family_identity", "levels", "value"},
            "observer archive native composite integral")
        try:
            value = float.fromhex(item["value"])
        except (TypeError, ValueError) as error:
            raise ValueError("observer archive native integral has invalid float.hex data") from error
        native_integrals.append(_NativeCompositeIntegral(
            Identity.from_token(item["family_identity"]), tuple(item["levels"]), value))
    snapshot = OutputSnapshot(
        _clock(row["clock"]),
        _provenance(row["provenance"]),
        tuple(_decode_geometry(item, arrays) for item in row["geometries"]),
        tuple(_decode_field(item, arrays) for item in row["fields"]),
        row["metadata"],
        diagnostics=tuple(_diagnostic(item) for item in row["diagnostics"]),
        _native_composite_integrals=tuple(native_integrals),
    )
    frame = ObserverFrame(snapshot, _request(row["request"]))
    evidence: dict[str, Any] = {}
    projection = _frame_projection(frame, lambda name, value: evidence.setdefault(
        name, array_evidence(value)))
    if projection != dict(row):
        raise ValueError("observer archive frame projection changed during reconstruction")
    return frame


def decode_observer_frame(payload: Any) -> ObserverFrame:
    """Decode and authenticate one complete observer-frame archive from exact bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("decode_observer_frame payload must be exact bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)) or _MANIFEST_MEMBER not in names:
                raise ValueError("observer archive has duplicate or missing members")
            if archive.comment:
                raise ValueError("observer archive ZIP comment must be empty")
            for info in infos:
                if info.is_dir() or info.compress_type != zipfile.ZIP_STORED \
                        or info.date_time != _ZIP_TIME or info.extra or info.comment \
                        or info.create_system != 3 \
                        or info.external_attr != _ZIP_EXTERNAL_ATTR \
                        or info.flag_bits != 0 or info.internal_attr != 0 \
                        or info.volume != 0 \
                        or info.extract_version not in {20, 45} \
                        or info.create_version not in {20, 45}:
                    raise ValueError("observer archive ZIP member metadata is not canonical")
            manifest_payload = archive.read(_MANIFEST_MEMBER)
            manifest = _decode_json(manifest_payload)
            row = _mapping(manifest, {
                "schema_version", "format", "frame_identity", "archive_identity", "frame",
                "arrays",
            }, "observer archive manifest")
            if row["schema_version"] != _SCHEMA_VERSION or row["format"] != _FORMAT:
                raise ValueError("observer archive schema/version is unsupported")
            arrays = _array_members(archive, row)
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError("observer archive container is corrupt") from error
    frame = _decode_frame(row["frame"], arrays)
    if frame.identity.token != row["frame_identity"]:
        raise ValueError("observer archive frame identity mismatch")
    evidence = {
        name: dict(description["evidence"])
        for name, description in row["arrays"].items()
    }
    expected = make_identity("observer-frame-archive", {
        "schema_version": _SCHEMA_VERSION,
        "frame": row["frame"],
        "arrays": evidence,
    })
    if expected.token != row["archive_identity"] \
            or observer_archive_identity(frame) != expected:
        raise ValueError("observer archive identity mismatch")
    return frame


def write_observer_archive(path: Any, frame: ObserverFrame, *, fsync: bool = True) -> str:
    """Create one no-clobber archive and return the SHA-256 digest of its exact file bytes."""
    if type(fsync) is not bool:
        raise TypeError("write_observer_archive fsync must be an exact bool")
    target = Path(path)
    payload = encode_observer_frame(frame)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    if fsync:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory = os.open(target.parent, flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return hashlib.sha256(payload).hexdigest()


def read_observer_archive(path: Any, *, expected_sha256: str | None = None) -> ObserverFrame:
    """Read one archive, optionally authenticating its complete file-byte SHA-256 first."""
    payload = Path(path).read_bytes()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) \
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise TypeError("expected_sha256 must be one lowercase SHA-256 digest or None")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("observer archive file SHA-256 mismatch")
    return decode_observer_frame(payload)


__all__ = [
    "decode_observer_frame", "encode_observer_frame", "observer_archive_identity",
    "read_observer_archive", "write_observer_archive",
]
