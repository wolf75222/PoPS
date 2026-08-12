"""Layer-neutral collective control plane for one shared restart checkpoint artifact.

The native Uniform/AMR codecs own field gathers, MPI transport and state mutation.  This module
only builds small deterministic control envelopes around one authenticated native communicator.
It never imports a Python MPI binding or executes a collective outside :mod:`pops._pops`.
"""

from __future__ import annotations

import os
import json
import math
import struct
import sys
import zipfile
from io import BytesIO
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from pops._native_collectives import (
    allgather_value,
    broadcast_bytes,
    broadcast_value,
    encode_value,
    rank as native_rank,
    require_world,
    size as native_size,
)


@dataclass(frozen=True, slots=True)
class CheckpointTopology:
    """Exact rank topology carried by one installed RuntimeInstance."""

    rank: int
    size: int
    communicator: Any = None

    @property
    def distributed(self) -> bool:
        return self.communicator is not None


@dataclass(frozen=True, slots=True)
class RootAttempt:
    """One root producer outcome with transport failure kept as a separate state."""

    value: Any = None
    producer_error: BaseException | None = None
    transport_error: BaseException | None = None


class InMemoryCheckpoint(Mapping[str, Any]):
    """Closed, object-free NPZ payload used by every restart rank.

    Every member is streamed through the bounded NPY reader while decoding the broadcast bytes.
    Native engine adapters therefore never reopen the shared checkpoint path and never retain a
    lazy ``NpzFile`` whose later access could perform rank-local filesystem I/O.
    """

    __slots__ = ("_arrays", "files")

    def __init__(self, arrays: Mapping[str, Any]) -> None:
        self._arrays = MappingProxyType(dict(arrays))
        self.files = tuple(self._arrays)

    def __getitem__(self, key: str) -> Any:
        return self._arrays[key]

    def __iter__(self):
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)


_CHECKPOINT_MAX_ARRAY_NDIM = 4  # component axis plus the compile-time native rank (1..3)
_MAX_SIZE_TEXT = len(str(sys.maxsize))
_NPY_HEADER_BUDGET = (
    len("{'descr': '', 'fortran_order': False, 'shape': (), }")
    + 32  # longest simple dtype spelling, including an exact Unicode/bytes item width
    + _CHECKPOINT_MAX_ARRAY_NDIM * (_MAX_SIZE_TEXT + 3)
    + 64  # NPY header alignment padding
    + 16  # magic, version and header-length words
)
_RESTART_TOKEN_CHARACTERS = len("pops.restart.v1:sha256:") + 64


@dataclass(frozen=True, slots=True)
class _NpyMemberPlan:
    name: str
    info: Any
    shape: tuple[int, ...]
    fortran_order: bool
    dtype: Any
    header_bytes: int
    data_bytes: int


def _checked_array_bytes(shape: Any, dtype: Any, *, name: str) -> int:
    if type(shape) is not tuple or len(shape) > _CHECKPOINT_MAX_ARRAY_NDIM:
        raise TypeError("checkpoint member %r exceeds the exact native array-rank bound" % name)
    elements = 1
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise TypeError("checkpoint member %r has an invalid array extent" % name)
        if extent > sys.maxsize or (extent and elements > sys.maxsize // extent):
            raise OverflowError("checkpoint member %r array shape exceeds size_t" % name)
        elements *= extent
    itemsize = int(dtype.itemsize)
    if itemsize < 0 or (itemsize and elements > sys.maxsize // itemsize):
        raise OverflowError("checkpoint member %r byte count exceeds size_t" % name)
    return elements * itemsize


def _require_checkpoint_dtype(dtype: Any, *, name: str) -> None:
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.itemsize <= 0
        or dtype.kind not in "biufcSU"
    ):
        raise TypeError("checkpoint member %r must use one primitive, object-free dtype" % name)


def _read_npy_header(archive: Any, info: Any, name: str) -> _NpyMemberPlan:
    import numpy as np

    with archive.open(info, "r") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                member, max_header_size=_NPY_HEADER_BUDGET
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                member, max_header_size=_NPY_HEADER_BUDGET
            )
        else:
            raise ValueError(
                "checkpoint member %r uses unsupported NPY version %r" % (name, version)
            )
        header_bytes = int(member.tell())
    dtype = np.dtype(dtype)
    _require_checkpoint_dtype(dtype, name=name)
    if fortran_order:
        raise ValueError(
            "checkpoint member %r must use the canonical C-contiguous NPY layout" % name
        )
    data_bytes = _checked_array_bytes(tuple(shape), dtype, name=name)
    if header_bytes > _NPY_HEADER_BUDGET or info.file_size != header_bytes + data_bytes:
        raise ValueError(
            "checkpoint member %r central-directory size differs from its NPY header" % name
        )
    return _NpyMemberPlan(
        name,
        info,
        tuple(shape),
        bool(fortran_order),
        dtype,
        header_bytes,
        data_bytes,
    )


def _read_npy_array(archive: Any, plan: _NpyMemberPlan) -> Any:
    import numpy as np

    with archive.open(plan.info, "r") as member:
        value = np.lib.format.read_array(
            member, allow_pickle=False, max_header_size=_NPY_HEADER_BUDGET
        )
        if member.read(1):
            raise ValueError("checkpoint member %r has trailing NPY bytes" % plan.name)
    array = np.asarray(value)
    if array.shape != plan.shape or array.dtype != plan.dtype:
        raise ValueError("checkpoint member %r changed after bounded extraction" % plan.name)
    array.setflags(write=False)
    return array


def _manifest_character_budget(names: tuple[str, ...]) -> int:
    """Derive the largest canonical manifest text from its exact member set and ND schema."""
    identity = {
        "domain": "semantic",
        "schema_version": sys.maxsize,
        "algorithm": "sha256",
        "hexdigest": "f" * 64,
    }
    evidence = {}
    for name in names:
        evidence[name] = {
            "dtype": "<U" + "9" * _MAX_SIZE_TEXT,
            "shape": [sys.maxsize] * _CHECKPOINT_MAX_ARRAY_NDIM,
            "content_sha256": "f" * 64,
        }
    maximal = {
        "schema_version": sys.maxsize,
        "runtime_kind": "multi_layout_uniform",
        "semantic_identity": identity,
        "artifact_identity": dict(identity, domain="artifact"),
        "bind_identity": dict(identity, domain="bind"),
        "run_identity": dict(identity, domain="run"),
        "clock": {"time": "-0x1.fffffffffffffp+1023", "macro_step": -sys.maxsize},
        "arrays": evidence,
        "restart_identity": dict(identity, domain="restart"),
    }
    return len(json.dumps(maximal, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _require_manifest_restart_identity(manifest: Mapping[str, Any], token: str) -> None:
    from pops._generated_release_contract import CHECKPOINT_ENVELOPE_SCHEMA_VERSION
    from pops.identity import Identity, make_identity

    def identity(field: str, domain: str) -> Identity:
        value = manifest[field]
        if not isinstance(value, Mapping) or set(value) != {
            "domain",
            "schema_version",
            "algorithm",
            "hexdigest",
        }:
            raise TypeError("checkpoint %s has an invalid exact identity schema" % field)
        digest = value["hexdigest"]
        if (
            value["domain"] != domain
            or isinstance(value["schema_version"], bool)
            or not isinstance(value["schema_version"], int)
            or value["algorithm"] != "sha256"
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
        ):
            raise ValueError("checkpoint %s has an invalid exact identity" % field)
        try:
            raw_digest = bytes.fromhex(digest)
        except ValueError:
            raise ValueError("checkpoint %s identity digest is not hexadecimal" % field) from None
        if raw_digest.hex() != digest:
            raise ValueError("checkpoint %s identity digest is not canonical" % field)
        return Identity(domain, value["schema_version"], "sha256", raw_digest)

    expected_keys = {
        "schema_version",
        "runtime_kind",
        "semantic_identity",
        "artifact_identity",
        "bind_identity",
        "run_identity",
        "clock",
        "arrays",
        "restart_identity",
    }
    if set(manifest) != expected_keys:
        raise ValueError("checkpoint manifest has an invalid exact schema")
    if (
        isinstance(manifest["schema_version"], bool)
        or manifest["schema_version"] != CHECKPOINT_ENVELOPE_SCHEMA_VERSION
        or not isinstance(manifest["runtime_kind"], str)
        or manifest["runtime_kind"] not in {"uniform", "amr", "multi_layout_uniform"}
    ):
        raise ValueError("checkpoint manifest version/runtime kind is unsupported")
    clock = manifest["clock"]
    if (
        not isinstance(clock, Mapping)
        or set(clock) != {"time", "macro_step"}
        or not isinstance(clock["time"], str)
        or isinstance(clock["macro_step"], bool)
        or not isinstance(clock["macro_step"], int)
    ):
        raise TypeError("checkpoint manifest clock has an invalid exact schema")
    try:
        accepted_time = float.fromhex(clock["time"])
    except ValueError:
        raise ValueError("checkpoint manifest clock time is not canonical hexadecimal") from None
    if not math.isfinite(accepted_time) or accepted_time.hex() != clock["time"]:
        raise ValueError("checkpoint manifest clock time is not canonical finite binary64")
    identity("semantic_identity", "semantic")
    identity("artifact_identity", "artifact")
    identity("bind_identity", "bind")
    identity("run_identity", "run")
    base = {key: manifest[key] for key in expected_keys - {"restart_identity"}}
    expected = make_identity("restart", base)
    recorded = identity("restart_identity", "restart")
    if recorded.token != expected.token or token != expected.token:
        raise ValueError("checkpoint manifest/restart identity is not authentic")


def _checkpoint_directory_authority(payload: bytes) -> tuple[int, int, int]:
    """Read the classic/ZIP64 EOCD bounds before ``ZipFile`` allocates directory rows."""
    signature = b"PK\x05\x06"
    search_start = max(0, len(payload) - (22 + 0xFFFF))
    search_end = len(payload)
    eocd = -1
    record = None
    while search_end > search_start:
        candidate = payload.rfind(signature, search_start, search_end)
        if candidate < 0:
            break
        if candidate + 22 <= len(payload):
            candidate_record = struct.unpack_from("<4s4H2LH", payload, candidate)
            if candidate + 22 + candidate_record[-1] == len(payload):
                eocd = candidate
                record = candidate_record
                break
        search_end = candidate
    if record is None:
        raise ValueError("checkpoint NPZ has no canonical end-of-directory record")
    (
        _,
        disk,
        directory_disk,
        disk_members,
        total_members,
        directory_size,
        directory_offset,
        comment_size,
    ) = record
    if disk != 0 or directory_disk != 0 or disk_members != total_members or comment_size != 0:
        raise ValueError("checkpoint NPZ must be one single-disk archive")
    if total_members != 0xFFFF and directory_size != 0xFFFFFFFF and directory_offset != 0xFFFFFFFF:
        if directory_offset + directory_size != eocd:
            raise ValueError("checkpoint NPZ central-directory bounds are not canonical")
        return int(total_members), int(directory_offset), int(directory_size)

    locator = eocd - 20
    if locator < 0 or payload[locator : locator + 4] != b"PK\x06\x07":
        raise ValueError("checkpoint ZIP64 archive lacks its canonical locator")
    _, zip64_disk, zip64_offset, disks = struct.unpack_from("<4sLQL", payload, locator)
    if zip64_disk != 0 or disks != 1 or zip64_offset > locator - 56:
        raise ValueError("checkpoint ZIP64 locator is invalid")
    if payload[zip64_offset : zip64_offset + 4] != b"PK\x06\x06":
        raise ValueError("checkpoint ZIP64 end-of-directory record is invalid")
    (
        _,
        record_size,
        _version_made,
        _version_needed,
        zip64_disk,
        zip64_directory_disk,
        zip64_disk_members,
        zip64_total_members,
        zip64_directory_size,
        zip64_directory_offset,
    ) = struct.unpack_from("<4sQ2H2L4Q", payload, zip64_offset)
    if (
        record_size < 44
        or zip64_offset + 12 + record_size > locator
        or zip64_disk != 0
        or zip64_directory_disk != 0
        or zip64_disk_members != zip64_total_members
        or zip64_directory_offset + zip64_directory_size != zip64_offset
    ):
        raise ValueError("checkpoint ZIP64 end-of-directory authority is invalid")
    return (
        int(zip64_total_members),
        int(zip64_directory_offset),
        int(zip64_directory_size),
    )


def _checkpoint_central_directory_preflight(payload: bytes, budget: Any) -> int:
    """Walk bounded central headers without constructing ``ZipInfo`` or member-name objects."""
    members, directory_offset, directory_size = _checkpoint_directory_authority(payload)
    if members <= 0 or members > budget.max_members:
        raise ValueError("checkpoint NPZ member count exceeds its live resource budget")
    directory_end = directory_offset + directory_size
    if directory_offset < 0 or directory_end > len(payload):
        raise ValueError("checkpoint NPZ central directory exceeds its byte envelope")
    position = directory_offset
    name_bytes = 0
    for _index in range(members):
        if position + 46 > directory_end or payload[position : position + 4] != b"PK\x01\x02":
            raise ValueError("checkpoint NPZ central directory has an invalid member header")
        flags = struct.unpack_from("<H", payload, position + 8)[0]
        compression = struct.unpack_from("<H", payload, position + 10)[0]
        filename_size, extra_size, comment_size, disk = struct.unpack_from(
            "<4H", payload, position + 28
        )
        if (
            flags & 1
            or compression != zipfile.ZIP_DEFLATED
            or filename_size <= 4
            or extra_size > 28
            or comment_size != 0
            or disk != 0
        ):
            raise ValueError("checkpoint NPZ central directory has an invalid member contract")
        name_bytes += int(filename_size)
        if name_bytes > budget.max_manifest_characters + 4 * members:
            raise ValueError("checkpoint NPZ member names exceed the live manifest resource budget")
        position += 46 + int(filename_size) + int(extra_size)
        if position > directory_end:
            raise ValueError("checkpoint NPZ central member exceeds its directory envelope")
    if position != directory_end:
        raise ValueError("checkpoint NPZ central directory has trailing data")
    return members


def _checkpoint_zip_members(
    payload: bytes, archive: Any, declared_members: int
) -> tuple[tuple[str, ...], dict[str, Any]]:
    infos = tuple(archive.infolist())
    if not infos or len(infos) != declared_members or len(infos) > len(payload) // 46:
        raise ValueError("checkpoint NPZ central directory has an invalid member count")
    names = []
    by_name = {}
    total_compressed = 0
    for info in infos:
        filename = info.filename
        if (
            not isinstance(filename, str)
            or not filename.endswith(".npy")
            or not filename[:-4]
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or info.is_dir()
            or info.flag_bits & 1
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.compress_size < 0
            or info.file_size < 0
        ):
            raise ValueError("checkpoint NPZ contains an invalid array member")
        name = filename[:-4]
        if name in by_name:
            raise ValueError("checkpoint NPZ contains duplicate array names")
        if info.compress_size > len(payload):
            raise ValueError("checkpoint NPZ member exceeds its sealed byte envelope")
        total_compressed += int(info.compress_size)
        if total_compressed > len(payload):
            raise ValueError("checkpoint NPZ compressed members exceed its byte envelope")
        names.append(name)
        by_name[name] = info
    return tuple(names), by_name


def decode_checkpoint_bytes(
    payload: Any, budget: Any, *, allow_reviewed_unsealed: bool = False
) -> InMemoryCheckpoint:
    """Preflight and stream one exact NPZ into bounded immutable arrays.

    The caller must supply the exact resource authority installed from the live runtime. Central
    directory sizes are rejected against it before any NPY header or manifest is materialized. The
    sealed manifest then authenticates and may only narrow those live allocation bounds.
    """
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("restart payload must be non-empty exact bytes")
    import numpy as np
    from pops._manifest_protocol import strict_json_loads
    from pops.runtime._checkpoint_resource_budget import CheckpointResourceBudget
    from pops.runtime._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY

    if type(budget) is not CheckpointResourceBudget:
        raise TypeError("restart decode requires one exact live checkpoint resource budget")
    if type(allow_reviewed_unsealed) is not bool:
        raise TypeError("restart unsealed-review policy must be an exact bool")
    if len(payload) > budget.max_archive_bytes:
        raise ValueError("checkpoint NPZ archive exceeds its live resource budget")
    declared_members = _checkpoint_central_directory_preflight(payload, budget)
    try:
        archive_context = zipfile.ZipFile(BytesIO(payload), "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("restart payload is not one exact NPZ archive") from error
    with archive_context as archive:
        names, infos = _checkpoint_zip_members(payload, archive, declared_members)
        if len(names) > budget.max_members:
            raise ValueError("checkpoint NPZ member count exceeds its live resource budget")
        name_characters = 0
        for name in names:
            name_characters += len(name)
            if name_characters > budget.max_manifest_characters:
                raise ValueError(
                    "checkpoint NPZ member names exceed the live manifest resource budget"
                )
        total_uncompressed = 0
        for info in infos.values():
            if info.file_size > budget.max_array_bytes + _NPY_HEADER_BUDGET:
                raise ValueError("checkpoint NPZ member exceeds its live per-array resource budget")
            total_uncompressed += int(info.file_size)
            if total_uncompressed > budget.max_uncompressed_bytes:
                raise ValueError(
                    "checkpoint NPZ uncompressed members exceed the live resource budget"
                )
        reserved = {MANIFEST_KEY, IDENTITY_KEY}
        present_reserved = reserved.intersection(names)
        if present_reserved and present_reserved != reserved:
            raise ValueError("checkpoint NPZ has a partial canonical envelope")

        plans = {}
        already_loaded = {}
        if present_reserved:
            payload_names = tuple(name for name in names if name not in reserved)
            manifest_plan = _read_npy_header(archive, infos[MANIFEST_KEY], MANIFEST_KEY)
            manifest_budget = _manifest_character_budget(payload_names)
            if (
                manifest_plan.shape != ()
                or manifest_plan.dtype.kind != "U"
                or manifest_plan.dtype.itemsize % 4
                or manifest_plan.dtype.itemsize // 4 > manifest_budget
                or manifest_plan.dtype.itemsize // 4 > budget.max_manifest_characters
                or manifest_plan.data_bytes > budget.max_array_bytes
            ):
                raise ValueError("checkpoint manifest member exceeds its exact schema budget")
            manifest_array = _read_npy_array(archive, manifest_plan)
            manifest_text = str(manifest_array.item())
            manifest = strict_json_loads(manifest_text, where="checkpoint manifest preflight")
            if not isinstance(manifest, Mapping) or not isinstance(manifest.get("arrays"), Mapping):
                raise TypeError("checkpoint manifest arrays must be an exact mapping")
            if manifest.get("runtime_kind") != budget.runtime_kind:
                raise ValueError(
                    "checkpoint manifest runtime kind differs from its live resource authority"
                )
            evidence = manifest["arrays"]
            if set(evidence) != set(payload_names):
                raise ValueError("checkpoint NPZ keys differ from its exact manifest")

            expected_data_bytes = 0
            expected = {}
            for name in payload_names:
                row = evidence[name]
                if not isinstance(row, Mapping) or set(row) != {
                    "dtype",
                    "shape",
                    "content_sha256",
                }:
                    raise TypeError("checkpoint manifest array evidence has an invalid schema")
                dtype_text = row["dtype"]
                shape_data = row["shape"]
                digest = row["content_sha256"]
                if not isinstance(dtype_text, str) or not isinstance(shape_data, list):
                    raise TypeError("checkpoint manifest dtype/shape evidence is invalid")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError("checkpoint manifest content digest is not canonical sha256")
                dtype = np.dtype(dtype_text)
                _require_checkpoint_dtype(dtype, name=name)
                shape = tuple(shape_data)
                data_bytes = _checked_array_bytes(shape, dtype, name=name)
                expected_data_bytes += data_bytes
                if (
                    data_bytes > budget.max_array_bytes
                    or expected_data_bytes > budget.max_uncompressed_bytes
                ):
                    raise ValueError("checkpoint manifest arrays exceed the live resource budget")
                expected[name] = (shape, dtype, data_bytes)

            identity_plan = _read_npy_header(archive, infos[IDENTITY_KEY], IDENTITY_KEY)
            if (
                identity_plan.shape != ()
                or identity_plan.dtype.kind != "U"
                or identity_plan.dtype.itemsize % 4
                or identity_plan.dtype.itemsize // 4 != _RESTART_TOKEN_CHARACTERS
            ):
                raise TypeError("checkpoint restart identity member must be scalar Unicode")
            identity_array = _read_npy_array(archive, identity_plan)
            restart_token = str(identity_array.item())
            if identity_plan.dtype.itemsize // 4 != len(restart_token):
                raise ValueError("checkpoint restart identity member has non-canonical padding")
            _require_manifest_restart_identity(manifest, restart_token)

            aggregate_budget = (
                manifest_plan.info.file_size
                + identity_plan.info.file_size
                + expected_data_bytes
                + len(payload_names) * _NPY_HEADER_BUDGET
            )
            if total_uncompressed > min(aggregate_budget, budget.max_uncompressed_bytes):
                raise ValueError(
                    "checkpoint NPZ uncompressed members exceed the authenticated manifest budget"
                )
            for name in payload_names:
                plan = _read_npy_header(archive, infos[name], name)
                shape, dtype, data_bytes = expected[name]
                if (
                    plan.shape != shape
                    or plan.dtype.str != dtype.str
                    or plan.data_bytes != data_bytes
                ):
                    raise ValueError(
                        "checkpoint member %r differs from its exact manifest shape/dtype" % name
                    )
                plans[name] = plan
            plans[MANIFEST_KEY] = manifest_plan
            plans[IDENTITY_KEY] = identity_plan
            already_loaded[MANIFEST_KEY] = manifest_array
            already_loaded[IDENTITY_KEY] = identity_array
        else:
            if not allow_reviewed_unsealed or not budget.authority.startswith(
                "reviewed-checkpoint-archive-sha256:"
            ):
                raise ValueError(
                    "unsealed historical checkpoint requires an exact reviewed migration authority"
                )
            for name in names:
                plans[name] = _read_npy_header(archive, infos[name], name)

        arrays = {}
        for name in names:
            arrays[name] = already_loaded.get(name)
            if arrays[name] is None:
                arrays[name] = _read_npy_array(archive, plans[name])
        return InMemoryCheckpoint(arrays)


def checkpoint_topology(owner: Any) -> CheckpointTopology:
    """Project an installed ExecutionContext without inferring a communicator."""
    context = getattr(owner, "_execution_context", None)
    resource = getattr(context, "communicator", None)
    identity = getattr(resource, "identity", None)
    handle = getattr(resource, "handle", None)
    if identity == "serial":
        if handle is not None:
            raise ValueError("serial checkpoint context hides a communicator handle")
        return CheckpointTopology(0, 1)
    if identity is None:
        raise ValueError(
            "checkpoint requires the authenticated ExecutionContext installed by pops.bind"
        )
    if identity != "MPI_COMM_WORLD":
        raise ValueError(
            "distributed checkpoint requires the authenticated MPI_COMM_WORLD "
            "ExecutionContext resource"
        )
    native = require_world(handle)
    return CheckpointTopology(native_rank(native), native_size(native), native)


def canonical_checkpoint_path(value: Any, *, extension: str = ".npz") -> Path:
    """Return one lexical absolute path suitable for rank-to-rank equality checks."""
    if not isinstance(extension, str) or not extension.startswith(".") or "/" in extension:
        raise TypeError("checkpoint extension must be a canonical file suffix")
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise TypeError("checkpoint path must be non-empty filesystem text")
    path = Path(text)
    if path.suffix != extension:
        path = path.with_suffix(extension)
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _error_record(error: BaseException) -> dict[str, str]:
    """Return one transport-safe error record without erasing its semantic family.

    MPI peers cannot re-raise another rank's exception object, but reducing every failure to a
    ``RuntimeError`` also destroys useful scientific contracts such as ``ValueError`` for an
    invalid checkpoint manifest.  Carry the exact qualified type for diagnostics and one closed
    builtin family for deterministic reconstruction on every rank.
    """
    if isinstance(error, FileNotFoundError):
        family = "FileNotFoundError"
    elif isinstance(error, PermissionError):
        family = "PermissionError"
    elif isinstance(error, OSError):
        family = "OSError"
    elif isinstance(error, TypeError):
        family = "TypeError"
    elif isinstance(error, ValueError):
        family = "ValueError"
    elif isinstance(error, KeyError):
        family = "KeyError"
    elif isinstance(error, IndexError):
        family = "IndexError"
    elif isinstance(error, NotImplementedError):
        family = "NotImplementedError"
    elif isinstance(error, AssertionError):
        family = "AssertionError"
    else:
        family = "RuntimeError"
    error_type = type(error)
    try:
        message = str(error)
    except BaseException:
        # Exception formatting is user-overridable. It must never fail before a peer enters the
        # matching broadcast/all-gather, otherwise one malformed rank-local error can deadlock the
        # remaining ranks. The exact qualified type still identifies the local cause.
        message = "<exception message unavailable>"
    return {
        "family": family,
        "type": "%s.%s" % (error_type.__module__, error_type.__qualname__),
        "message": message,
    }


_ERROR_FAMILIES: dict[str, type[BaseException]] = {
    "AssertionError": AssertionError,
    "FileNotFoundError": FileNotFoundError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "NotImplementedError": NotImplementedError,
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
}


def _validated_error_record(value: Any, *, phase: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"family", "type", "message"}:
        raise RuntimeError("checkpoint %s returned an invalid error record" % phase)
    if any(not isinstance(value[key], str) for key in ("family", "type", "message")):
        raise RuntimeError("checkpoint %s returned a non-text error record" % phase)
    if value["family"] not in _ERROR_FAMILIES or not value["type"]:
        raise RuntimeError("checkpoint %s returned an unknown error family" % phase)
    return value


def _raise_collective_failure(
    phase: str,
    failures: tuple[tuple[int, Mapping[str, str]], ...],
) -> None:
    """Raise one deterministic builtin family while retaining every rank-local cause."""
    if not failures:
        return
    families = {record["family"] for _rank, record in failures}
    error_type = _ERROR_FAMILIES[next(iter(families))] if len(families) == 1 else RuntimeError
    details = "; ".join(
        "rank %d: %s: %s" % (rank, record["type"], record["message"]) for rank, record in failures
    )
    raise error_type("collective checkpoint %s failed: %s" % (phase, details))


def root_value(
    topology: CheckpointTopology,
    phase: str,
    producer: Callable[[], Any],
) -> Any:
    """Run one Python/filesystem decision on rank zero and broadcast its result or failure."""
    if not isinstance(phase, str) or not phase:
        raise TypeError("checkpoint phase must be non-empty text")
    if not callable(producer):
        raise TypeError("checkpoint root producer must be callable")
    envelope = None
    if topology.rank == 0:
        try:
            envelope = {"value": producer(), "error": None}
        except BaseException as error:
            if not topology.distributed:
                raise
            envelope = {"value": None, "error": _error_record(error)}
    if topology.distributed:
        envelope = broadcast_value(topology.communicator, envelope, root=0)
    else:
        # Keep serial and MPI semantics identical: generic control envelopes never carry bulk
        # bytes, even though no transport would otherwise force their encoding in serial.
        encode_value(envelope)
    if not isinstance(envelope, Mapping) or set(envelope) != {"value", "error"}:
        raise RuntimeError("checkpoint %s broadcast returned an invalid envelope" % phase)
    if envelope["error"] is not None:
        record = _validated_error_record(envelope["error"], phase=phase)
        _raise_collective_failure(phase, ((0, record),))
    return envelope["value"]


def root_attempt(
    topology: CheckpointTopology,
    phase: str,
    producer: Callable[[], Any],
) -> RootAttempt:
    """Run one root producer without conflating its failure with broadcast transport.

    Callers that own rank-zero filesystem state can safely decide whether another collective is
    legal: a producer failure means the first transport completed, while ``transport_error`` means
    only rank zero may perform local compensation.
    """
    if not isinstance(phase, str) or not phase:
        raise TypeError("checkpoint phase must be non-empty text")
    if not callable(producer):
        raise TypeError("checkpoint root producer must be callable")
    envelope = None
    local_error = None
    if topology.rank == 0:
        try:
            envelope = {"value": producer(), "error": None}
        except BaseException as error:
            local_error = error
            envelope = {
                "value": None,
                "error": None if not topology.distributed else _error_record(error),
            }
    if not topology.distributed:
        if local_error is not None:
            return RootAttempt(producer_error=local_error)
        try:
            encode_value(envelope)
        except BaseException as error:
            return RootAttempt(transport_error=error)
    else:
        try:
            envelope = broadcast_value(topology.communicator, envelope, root=0)
        except BaseException as error:
            return RootAttempt(producer_error=local_error, transport_error=error)
    try:
        if not isinstance(envelope, Mapping) or set(envelope) != {"value", "error"}:
            raise RuntimeError("checkpoint %s broadcast returned an invalid envelope" % phase)
        if envelope["error"] is not None:
            record = _validated_error_record(envelope["error"], phase=phase)
            try:
                _raise_collective_failure(phase, ((0, record),))
            except BaseException as error:
                return RootAttempt(
                    producer_error=(
                        local_error if topology.rank == 0 and local_error is not None else error
                    )
                )
            raise AssertionError("checkpoint producer failure reconstruction returned")
    except BaseException as error:
        return RootAttempt(transport_error=error)
    return RootAttempt(value=envelope["value"])


def root_effect(
    topology: CheckpointTopology,
    phase: str,
    operation: Callable[[], Any],
) -> Any:
    """Run rank-zero file I/O while broadcasting only completion/failure, never bulk data."""
    if not callable(operation):
        raise TypeError("checkpoint root operation must be callable")
    result = None
    failure = None
    if topology.rank == 0:
        try:
            result = operation()
        except BaseException as error:
            if not topology.distributed:
                raise
            failure = _error_record(error)
    if topology.distributed:
        failure = broadcast_value(topology.communicator, failure, root=0)
    if failure is not None:
        record = _validated_error_record(failure, phase=phase)
        _raise_collective_failure(phase, ((0, record),))
    return result


def _bounded_checkpoint_path_bytes(path: Path, max_bytes: int) -> bytes:
    if not isinstance(path, Path) or isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("checkpoint bounded read requires an exact path and byte capacity")
    if max_bytes <= 0:
        raise ValueError("checkpoint bounded read requires a positive byte capacity")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError("checkpoint archive size exceeds its live resource budget")
    with path.open("rb") as stream:
        payload = stream.read(size)
        trailing = stream.read(1)
    if len(payload) != size or trailing:
        raise ValueError("checkpoint archive changed during its bounded read")
    return payload


def _bounded_checkpoint_stream_bytes(stream: Any, max_bytes: int) -> bytes:
    """Read one already-authenticated file descriptor without an unbounded allocation."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise TypeError("checkpoint bounded stream read requires a positive exact byte capacity")
    read = getattr(stream, "read", None)
    if not callable(read):
        raise TypeError("checkpoint bounded stream read requires a binary stream")
    payload = read(max_bytes)
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("checkpoint bounded stream read returned no exact bytes")
    trailing = read(1)
    if not isinstance(trailing, bytes):
        raise TypeError("checkpoint bounded stream read returned non-byte trailing data")
    if trailing:
        raise ValueError("checkpoint archive exceeds its live resource budget")
    return payload


def root_bytes(
    topology: CheckpointTopology,
    phase: str,
    producer: Callable[[], bytes],
    *,
    max_bytes: int,
) -> bytes:
    """Read bytes on rank zero and broadcast them directly through the native C++ transport."""
    if not isinstance(phase, str) or not phase:
        raise TypeError("checkpoint phase must be non-empty text")
    if not callable(producer):
        raise TypeError("checkpoint root byte producer must be callable")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise TypeError("checkpoint root byte broadcast requires a positive exact byte capacity")
    payload = b""
    failure = None
    if topology.rank == 0:
        try:
            payload = producer()
            if not isinstance(payload, bytes) or not payload:
                raise TypeError("checkpoint root byte producer must return non-empty exact bytes")
            if len(payload) > max_bytes:
                raise ValueError("checkpoint root bytes exceed their live resource budget")
        except BaseException as error:
            if not topology.distributed:
                raise
            failure = _error_record(error)
    if topology.distributed:
        failure = broadcast_value(topology.communicator, failure, root=0)
    if failure is not None:
        record = _validated_error_record(failure, phase=phase)
        _raise_collective_failure(phase, ((0, record),))
    if topology.distributed:
        declared_size = broadcast_value(
            topology.communicator,
            len(payload) if topology.rank == 0 else None,
            root=0,
        )
        size_error = None
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or declared_size <= 0
            or declared_size > max_bytes
        ):
            size_error = ValueError("checkpoint broadcast size exceeds its live resource budget")
        consensus(topology, "%s byte-budget preflight" % phase, error=size_error)
        payload = broadcast_bytes(topology.communicator, payload, root=0)
        if len(payload) != declared_size:
            raise ValueError("checkpoint broadcast bytes differ from their declared size")
    if len(payload) > max_bytes:
        raise ValueError("checkpoint broadcast bytes exceed their live resource budget")
    return payload


def consensus(
    topology: CheckpointTopology,
    phase: str,
    *,
    error: BaseException | None = None,
    value: Any = None,
) -> tuple[Mapping[str, Any], ...]:
    """Convert every local phase result into one ordered all-rank decision.

    The next collective phase may start only after this function returns successfully.  In serial,
    the original exception type is preserved. In MPI, all ranks raise the same deterministic
    builtin semantic family and text; mixed families become ``RuntimeError`` while the message
    retains every rank-local exact exception type.
    """
    if error is not None and not isinstance(error, BaseException):
        raise TypeError("checkpoint consensus error must be an exception or None")
    if not topology.distributed:
        if error is not None:
            raise error
        return ({"rank": 0, "value": value, "error": None},)
    envelope = {
        "rank": topology.rank,
        "value": value,
        "error": None if error is None else _error_record(error),
    }
    rows = allgather_value(topology.communicator, envelope)
    if len(rows) != topology.size:
        raise RuntimeError(
            "checkpoint %s consensus returned %d ranks, expected %d"
            % (phase, len(rows), topology.size)
        )
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"rank", "value", "error"}:
            raise RuntimeError("checkpoint %s consensus row has an invalid schema" % phase)
        if row["rank"] != index:
            raise RuntimeError("checkpoint %s consensus rank order is invalid" % phase)
        normalized.append(row)
    failures = tuple(
        (
            int(row["rank"]),
            _validated_error_record(row["error"], phase=phase),
        )
        for row in normalized
        if row["error"] is not None
    )
    if failures:
        _raise_collective_failure(phase, failures)
    return tuple(normalized)


def collective_checkpoint_capture(
    owner: Any,
    phase_prefix: str,
    prepare: Callable[[], tuple[Any, str]],
    capture: Callable[[Any], tuple[Any, str]],
    publish: Callable[[Any], Any],
) -> Any:
    """Run a deadlock-safe checkpoint capture and rank-zero publication.

    ``prepare`` must be collective-free and return the opaque plan plus its content identity.
    Every rank agrees on that identity before ``capture`` may invoke its first native collective.
    ``capture`` returns an in-memory sealed artifact plus its restart identity; every rank agrees on
    that identity before ``publish`` performs any filesystem write on rank zero.
    """
    if not isinstance(phase_prefix, str) or not phase_prefix:
        raise TypeError("checkpoint capture phase prefix must be non-empty text")
    if not all(callable(callback) for callback in (prepare, capture, publish)):
        raise TypeError("checkpoint capture callbacks must be callable")
    topology = checkpoint_topology(owner)

    plan = None
    plan_identity = None
    prepare_error = None
    try:
        prepared = prepare()
        if type(prepared) is not tuple or len(prepared) != 2:
            raise TypeError("checkpoint prepare must return (plan, identity)")
        plan, plan_identity = prepared
        if not isinstance(plan_identity, str) or not plan_identity:
            raise TypeError("checkpoint capture-plan identity must be non-empty text")
    except BaseException as error:
        prepare_error = error
    rows = consensus(
        topology,
        "%s preflight" % phase_prefix,
        error=prepare_error,
        value=plan_identity,
    )
    if any(row["value"] != plan_identity for row in rows):
        raise RuntimeError(
            "collective checkpoint %s capture plans differ across ranks" % phase_prefix
        )

    artifact = None
    artifact_identity = None
    capture_error = None
    try:
        captured = capture(plan)
        if type(captured) is not tuple or len(captured) != 2:
            raise TypeError("checkpoint capture must return (artifact, identity)")
        artifact, artifact_identity = captured
        if not isinstance(artifact_identity, str) or not artifact_identity:
            raise TypeError("sealed checkpoint identity must be non-empty text")
    except BaseException as error:
        capture_error = error
    rows = consensus(
        topology,
        "%s sealed payload" % phase_prefix,
        error=capture_error,
        value=artifact_identity,
    )
    if any(row["value"] != artifact_identity for row in rows):
        raise RuntimeError(
            "collective checkpoint %s sealed payloads differ across ranks" % phase_prefix
        )

    return root_value(
        topology,
        "%s publication" % phase_prefix,
        lambda: publish(artifact),
    )


def _result_evidence(value: Any) -> Any:
    from pops.identity import Identity

    # Identity.to_data() is the canonical CBOR form and intentionally carries a raw 32-byte digest.
    # Structured MPI collectives deliberately refuse bytes so binary payloads cannot accidentally
    # bypass the direct byte transport.  Restart consensus needs only equality evidence, for which
    # the lossless, domain-qualified printable token is the exact representation.
    if type(value) is Identity:
        return value.token
    to_data = getattr(value, "to_data", None)
    return to_data() if callable(to_data) else value


def require_restart_bit_identical(value: Any, *, where: str) -> bool:
    """Require the exact restart reproducibility policy at every private handoff."""
    if type(value) is not bool:
        raise TypeError("%s bit_identical must be an exact bool" % where)
    return value


_RESTART_HIERARCHY_MODES = {
    "restore_recorded_hierarchy",
    "regrid_on_restart",
}


def require_restart_hierarchy_mode(value: Any, *, where: str) -> str:
    """Require one exact built-in hierarchy policy at every collective handoff."""
    if not isinstance(value, str) or value not in _RESTART_HIERARCHY_MODES:
        raise ValueError("%s hierarchy mode is unsupported" % where)
    return value


def restore_checkpoint_payload(
    owner: Any,
    executor: Any,
    payload: bytes,
    *,
    bit_identical: bool,
    hierarchy_mode: str = "restore_recorded_hierarchy",
    hierarchy_identity: str | None = None,
    phase_prefix: str = "native restart",
    after_native_apply: Callable[[Any], Any] | None = None,
    rollback_after_native_apply: Callable[[], Any] | None = None,
) -> Any:
    """Preflight and atomically apply one in-memory payload on the installed communicator.

    Every fallible preparation finishes with an all-rank consensus before the first native write.
    The accepted native snapshot remains rollback-capable through the apply and commit consensuses;
    only the final, non-fallible release discards it.  A RuntimeInstance may use the optional
    callback pair to publish its coupled Python envelope inside that same boundary.  Direct-engine
    callers do not provide either callback and retain their existing protocol exactly.
    """
    if not isinstance(phase_prefix, str) or not phase_prefix:
        raise TypeError("restart phase prefix must be non-empty text")
    if (after_native_apply is None) != (rollback_after_native_apply is None):
        raise TypeError(
            "restart outer-state publication requires both apply and rollback callbacks"
        )
    if after_native_apply is not None and not callable(after_native_apply):
        raise TypeError("restart outer-state apply callback must be callable")
    if rollback_after_native_apply is not None and not callable(rollback_after_native_apply):
        raise TypeError("restart outer-state rollback callback must be callable")
    topology = checkpoint_topology(owner)
    policy = None
    selected_hierarchy_mode = None
    selected_hierarchy_identity = None
    policy_error = None
    try:
        policy = require_restart_bit_identical(bit_identical, where="restart preparation policy")
        selected_hierarchy_mode = require_restart_hierarchy_mode(
            hierarchy_mode, where="restart preparation policy"
        )
        if selected_hierarchy_mode == "regrid_on_restart":
            from pops.identity import Identity

            selected = Identity.from_token(hierarchy_identity)
            if selected.domain != "restart-hierarchy":
                raise ValueError("restart preparation hierarchy identity has the wrong domain")
            selected_hierarchy_identity = selected.token
        elif hierarchy_identity is not None:
            raise ValueError(
                "restart preparation hierarchy identity is only valid with RegridOnRestart"
            )
    except BaseException as error:
        policy_error = error
    policy_rows = consensus(
        topology,
        "%s policy" % phase_prefix,
        error=policy_error,
        value={
            "bit_identical": policy,
            "hierarchy_mode": selected_hierarchy_mode,
            "hierarchy_identity": selected_hierarchy_identity,
        },
    )
    if any(row["value"] != policy_rows[0]["value"] for row in policy_rows[1:]):
        raise ValueError("%s restart policy differs across ranks" % phase_prefix)
    if policy and selected_hierarchy_mode == "regrid_on_restart":
        raise ValueError("%s cannot combine bit_identical=True with RegridOnRestart" % phase_prefix)
    method_names = (
        "_prepare_checkpoint_restart",
        "_begin_checkpoint_restart",
        "_apply_checkpoint_restart",
        "_commit_checkpoint_restart",
        "_finalize_checkpoint_restart",
        "_rollback_checkpoint_restart",
    )
    methods: dict[str, Callable[..., Any]] = {}
    protocol_error = None
    try:
        missing = []
        for name in method_names:
            method = getattr(executor, name, None)
            if not callable(method):
                missing.append(name)
                continue
            methods[name] = method
        if missing:
            raise TypeError(
                "restart engine lacks the exact in-memory transaction protocol: %s"
                % ", ".join(missing)
            )
    except BaseException as error:
        protocol_error = error
    consensus(
        topology,
        "%s provider protocol" % phase_prefix,
        error=protocol_error,
        value="|".join(method_names) if protocol_error is None else None,
    )

    prepared = None
    prepare_error = None
    try:
        if selected_hierarchy_mode == "regrid_on_restart":
            prepared = methods["_prepare_checkpoint_restart"](
                payload,
                bit_identical=policy,
                hierarchy_mode=selected_hierarchy_mode,
                hierarchy_identity=selected_hierarchy_identity,
            )
        else:
            prepared = methods["_prepare_checkpoint_restart"](payload, bit_identical=policy)
    except BaseException as error:
        prepare_error = error
    consensus(topology, "%s preflight" % phase_prefix, error=prepare_error)

    active = False
    begin_error = None
    try:
        methods["_begin_checkpoint_restart"]()
        active = True
    except BaseException as error:
        begin_error = error
    try:
        consensus(topology, "%s transaction begin" % phase_prefix, error=begin_error)
    except BaseException as original:
        rollback_error = None
        if active:
            try:
                methods["_rollback_checkpoint_restart"]()
            except BaseException as error:
                rollback_error = error
        try:
            consensus(
                topology,
                "%s begin rollback" % phase_prefix,
                error=rollback_error,
            )
        except BaseException as cleanup:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note("restart begin rollback also failed: %s" % cleanup)
        raise

    def rollback_after(original: BaseException, phase: str) -> None:
        rollback_error = None
        try:
            methods["_rollback_checkpoint_restart"]()
        except BaseException as error:
            rollback_error = error
        try:
            consensus(
                topology,
                "%s %s rollback" % (phase_prefix, phase),
                error=rollback_error,
            )
        except BaseException as cleanup:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note("restart rollback also failed: %s" % cleanup)

    result = None
    apply_error = None
    try:
        result = methods["_apply_checkpoint_restart"](prepared)
    except BaseException as error:
        apply_error = error
    try:
        rows = consensus(
            topology,
            "%s apply" % phase_prefix,
            error=apply_error,
            value=_result_evidence(result),
        )
        if any(row["value"] != rows[0]["value"] for row in rows[1:]):
            raise RuntimeError("%s ranks returned divergent restart evidence" % phase_prefix)
    except BaseException as original:
        rollback_after(original, "apply")
        raise

    outer_active = after_native_apply is not None

    def rollback_outer_after(original: BaseException, phase: str) -> None:
        outer_error = None
        if outer_active:
            try:
                cast(Callable[[], Any], rollback_after_native_apply)()
            except BaseException as error:
                outer_error = error
        try:
            consensus(
                topology,
                "%s %s outer rollback" % (phase_prefix, phase),
                error=outer_error,
            )
        except BaseException as cleanup:
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note("restart outer-state rollback also failed: %s" % cleanup)

    if outer_active:
        outer_error = None
        try:
            cast(Callable[[Any], Any], after_native_apply)(result)
        except BaseException as error:
            outer_error = error
        try:
            consensus(
                topology,
                "%s outer-state publication" % phase_prefix,
                error=outer_error,
            )
        except BaseException as original:
            rollback_outer_after(original, "outer-state publication")
            rollback_after(original, "outer-state publication")
            raise

    commit_error = None
    try:
        methods["_commit_checkpoint_restart"]()
    except BaseException as error:
        commit_error = error
    try:
        consensus(topology, "%s commit" % phase_prefix, error=commit_error)
    except BaseException as original:
        rollback_outer_after(original, "commit")
        rollback_after(original, "commit")
        raise

    # Finalization only releases snapshots that every rank has already agreed to commit.  Providers
    # must implement it as a no-throw release; the consensus turns a contract violation into one
    # coherent failure instead of allowing a peer to enter the next operation silently.
    finalize_error = None
    try:
        methods["_finalize_checkpoint_restart"]()
    except BaseException as error:
        finalize_error = error
    consensus(topology, "%s finalize" % phase_prefix, error=finalize_error)
    return result


def restore_checkpoint_path(
    owner: Any,
    executor: Any,
    path: Any,
    *,
    bit_identical: bool,
    hierarchy_mode: str = "restore_recorded_hierarchy",
    hierarchy_identity: str | None = None,
    phase_prefix: str = "native restart",
) -> Any:
    """Collectively read and restore one shared checkpoint through native transports.

    This is the direct-engine counterpart of ``RestartV3``.  It deliberately carries no
    ``ConsumerGraph`` cursor state, but it uses the same rank-zero read, exact path agreement,
    byte broadcast and all-rank transactional restore as a bound ``RuntimeInstance``.
    """
    if not isinstance(phase_prefix, str) or not phase_prefix:
        raise TypeError("restart phase prefix must be non-empty text")
    policy = require_restart_bit_identical(bit_identical, where="restart path policy")
    selected_hierarchy_mode = require_restart_hierarchy_mode(
        hierarchy_mode, where="restart path policy"
    )
    if selected_hierarchy_mode != "regrid_on_restart" and hierarchy_identity is not None:
        raise ValueError("restart path hierarchy identity is only valid with RegridOnRestart")
    topology = checkpoint_topology(owner)
    target = None
    target_text = None
    target_error = None
    try:
        target = canonical_checkpoint_path(path)
        target_text = str(target)
    except BaseException as error:
        target_error = error
    rows = consensus(
        topology,
        "%s target" % phase_prefix,
        error=target_error,
        value=target_text,
    )
    if target is None or target_text is None:
        raise RuntimeError("%s target consensus lost its local path" % phase_prefix)
    if any(row["value"] != target_text for row in rows):
        raise ValueError("%s target differs across ranks" % phase_prefix)
    from pops.runtime._checkpoint_resource_budget import require_checkpoint_resource_budget

    archive_budget = require_checkpoint_resource_budget(executor).max_archive_bytes
    payload = root_bytes(
        topology,
        "%s read" % phase_prefix,
        lambda: _bounded_checkpoint_path_bytes(target, archive_budget),
        max_bytes=archive_budget,
    )
    if selected_hierarchy_mode == "regrid_on_restart":
        return restore_checkpoint_payload(
            owner,
            executor,
            payload,
            bit_identical=policy,
            hierarchy_mode=selected_hierarchy_mode,
            hierarchy_identity=hierarchy_identity,
            phase_prefix=phase_prefix,
        )
    return restore_checkpoint_payload(
        owner,
        executor,
        payload,
        bit_identical=policy,
        phase_prefix=phase_prefix,
    )


__all__ = [
    "CheckpointTopology",
    "InMemoryCheckpoint",
    "RootAttempt",
    "canonical_checkpoint_path",
    "checkpoint_topology",
    "collective_checkpoint_capture",
    "consensus",
    "decode_checkpoint_bytes",
    "require_restart_bit_identical",
    "require_restart_hierarchy_mode",
    "restore_checkpoint_path",
    "restore_checkpoint_payload",
    "root_effect",
    "root_bytes",
    "root_attempt",
    "root_value",
]
