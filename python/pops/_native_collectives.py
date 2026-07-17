"""Strict value transport over the native PoPS MPI communicator.

MPI is owned entirely by the compiled :mod:`pops._pops` extension.  This module only projects
small Python control values to deterministic bytes before calling the native byte collectives; it
does not import an MPI Python binding, inspect process-global MPI state, or execute a collective in
Python.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


_WORLD_IDENTITY = "MPI_COMM_WORLD"
_FRAME_OK = b"\x00"
_FRAME_ERROR = b"\x01"


def _error_text(error: BaseException) -> bytes:
    return ("%s: %s" % (type(error).__name__, error)).encode(
        "utf-8", errors="backslashreplace"
    )


def _value_frame(value: Any) -> bytes:
    try:
        return _FRAME_OK + encode_value(value)
    except BaseException as error:
        return _FRAME_ERROR + _error_text(error)


def _bytes_frame(payload: Any) -> bytes:
    try:
        if not isinstance(payload, bytes):
            raise TypeError("native broadcast payload must be exact bytes")
        return _FRAME_OK + payload
    except BaseException as error:
        return _FRAME_ERROR + _error_text(error)


def _frame_error(frame: Any, *, where: str) -> str | None:
    if not isinstance(frame, bytes) or not frame:
        return "%s returned a malformed native frame" % where
    if frame.startswith(_FRAME_OK):
        return None
    if frame.startswith(_FRAME_ERROR):
        try:
            return frame[1:].decode("utf-8")
        except UnicodeDecodeError:
            return "%s returned a malformed error frame" % where
    return "%s returned an unknown native frame" % where


def _encode_node(value: Any, *, path: str) -> list[Any]:
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, int) and not isinstance(value, bool):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("native collective value at %s contains a non-finite float" % path)
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        raise TypeError(
            "native structured collectives refuse bytes at %s; use the direct byte transport"
            % path
        )
    if isinstance(value, tuple):
        return ["tuple", [
            _encode_node(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]]
    if isinstance(value, list):
        return ["list", [
            _encode_node(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]]
    if isinstance(value, Mapping):
        rows = []
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "native collective mapping at %s requires non-empty string keys" % path
                )
            rows.append([key, _encode_node(value[key], path="%s.%s" % (path, key))])
        return ["map", rows]
    raise TypeError(
        "native collective value at %s cannot encode %s"
        % (path, type(value).__name__)
    )


def _decode_node(node: Any, *, path: str) -> Any:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("native collective payload at %s has an invalid node" % path)
    tag = node[0]
    if tag == "none" and node == ["none"]:
        return None
    if tag == "bool" and len(node) == 2 and type(node[1]) is bool:
        return node[1]
    if tag == "int" and len(node) == 2 and isinstance(node[1], str):
        try:
            value = int(node[1], 10)
        except ValueError as exc:
            raise ValueError("native collective integer at %s is invalid" % path) from exc
        if str(value) != node[1]:
            raise ValueError("native collective integer at %s is not canonical" % path)
        return value
    if tag == "float" and len(node) == 2 and isinstance(node[1], str):
        try:
            value = float.fromhex(node[1])
        except ValueError as exc:
            raise ValueError("native collective float at %s is invalid" % path) from exc
        if not math.isfinite(value) or value.hex() != node[1]:
            raise ValueError("native collective float at %s is not canonical" % path)
        return value
    if tag == "str" and len(node) == 2 and isinstance(node[1], str):
        return node[1]
    if tag in {"tuple", "list"} and len(node) == 2 and isinstance(node[1], list):
        values = [
            _decode_node(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(node[1])
        ]
        return tuple(values) if tag == "tuple" else values
    if tag == "map" and len(node) == 2 and isinstance(node[1], list):
        result = {}
        previous = None
        for index, row in enumerate(node[1]):
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0]
            ):
                raise ValueError(
                    "native collective mapping row at %s[%d] is invalid" % (path, index)
                )
            key = row[0]
            if previous is not None and key <= previous:
                raise ValueError("native collective mapping at %s is not canonical" % path)
            previous = key
            result[key] = _decode_node(row[1], path="%s.%s" % (path, key))
        return result
    raise ValueError("native collective payload at %s has an unknown node" % path)


def encode_value(value: Any) -> bytes:
    """Encode one strict control-plane value to deterministic UTF-8 bytes."""
    text = json.dumps(
        _encode_node(value, path="$"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def decode_value(payload: Any) -> Any:
    """Decode and re-authenticate one strict control-plane payload."""
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("native collective payload must be non-empty exact bytes")
    try:
        node = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("native collective payload is not valid strict JSON") from exc
    value = _decode_node(node, path="$")
    if encode_value(value) != payload:
        raise ValueError("native collective payload is not canonical")
    return value


def require_world(communicator: Any) -> Any:
    """Require the exact active communicator object produced by ``_pops.mpi_world()``."""
    from pops import _pops

    if type(communicator) is not _pops._NativeWorldCommunicator:
        raise TypeError("distributed execution requires the native PoPS world communicator")
    if communicator.identity != _WORLD_IDENTITY or communicator.active is not True:
        raise ValueError("distributed execution requires active native MPI_COMM_WORLD")
    rank, size = int(communicator.rank), int(communicator.size)
    if size < 1 or rank < 0 or rank >= size:
        raise ValueError("native MPI_COMM_WORLD has an invalid rank topology")
    return communicator


def rank(communicator: Any) -> int:
    return int(require_world(communicator).rank)


def size(communicator: Any) -> int:
    return int(require_world(communicator).size)


def barrier(communicator: Any) -> None:
    require_world(communicator).barrier()


def broadcast_value(communicator: Any, value: Any, *, root: int = 0) -> Any:
    native = require_world(communicator)
    frame = _value_frame(value) if int(native.rank) == root else b""
    result = native.broadcast_bytes(frame, root)
    failure = _frame_error(result, where="native broadcast")
    if failure is not None:
        raise RuntimeError("native broadcast failed on rank %d: %s" % (root, failure))
    return decode_value(result[1:])


def broadcast_bytes(communicator: Any, payload: bytes, *, root: int = 0) -> bytes:
    """Broadcast one opaque payload directly in C++ without Python object serialization."""
    native = require_world(communicator)
    frame = _bytes_frame(payload) if int(native.rank) == root else b""
    result = native.broadcast_bytes(frame, root)
    failure = _frame_error(result, where="native byte broadcast")
    if failure is not None:
        raise RuntimeError("native byte broadcast failed on rank %d: %s" % (root, failure))
    return result[1:]


def allgather_value(communicator: Any, value: Any) -> tuple[Any, ...]:
    native = require_world(communicator)
    frames = native.allgather_bytes(_value_frame(value))
    if type(frames) is not tuple or len(frames) != int(native.size):
        raise RuntimeError("native allgather returned an invalid rank payload set")
    failures = [
        "rank %d: %s" % (owner, failure)
        for owner, frame in enumerate(frames)
        if (failure := _frame_error(frame, where="native allgather")) is not None
    ]
    if failures:
        raise RuntimeError("native allgather encoding failed: " + "; ".join(failures))
    return tuple(decode_value(frame[1:]) for frame in frames)


def allgather_bytes(communicator: Any, payload: bytes) -> tuple[bytes, ...]:
    """All-gather one opaque payload per rank through the native C++ byte collective."""
    native = require_world(communicator)
    frames = native.allgather_bytes(_bytes_frame(payload))
    if type(frames) is not tuple or len(frames) != int(native.size):
        raise RuntimeError("native byte allgather returned an invalid rank payload set")
    failures = [
        "rank %d: %s" % (owner, failure)
        for owner, frame in enumerate(frames)
        if (failure := _frame_error(frame, where="native byte allgather")) is not None
    ]
    if failures:
        raise RuntimeError("native byte allgather failed: " + "; ".join(failures))
    return tuple(frame[1:] for frame in frames)


__all__ = [
    "allgather_bytes", "allgather_value", "barrier", "broadcast_bytes", "broadcast_value",
    "decode_value", "encode_value", "rank", "require_world", "size",
]
