"""Inert exact-data contracts shared by checkpoint output and live runtimes.

This module intentionally depends on neither runtime implementation nor native extension.
It owns the concrete resource-budget type and canonical envelope member names so output
providers can enforce their already-installed allocation authority without importing
``pops.runtime``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
import sys
from collections.abc import Mapping
from typing import Any, Callable

from pops._program_resource_plan_contract import ProgramResourcePlanCapacityAuthority


MANIFEST_KEY = "pops_checkpoint_manifest"
IDENTITY_KEY = "pops_restart_identity"

# The persistent Program image is an opaque byte member in the Uniform/AMR NPZ envelope.  These
# scalar members are deliberately redundant: they make the lossless schema and plan identity
# visible to the bounded archive budget and to offline manifest readers without asking them to
# parse the C++ carrier first.  The carrier remains the source of truth and is authenticated again
# below before a restart can prepare a native restore.
PROGRAM_PERSISTENT_CHECKPOINT_KEY = "program_persistent_value_checkpoint"
PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY = "program_persistent_value_checkpoint_schema"
PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY = "program_resource_plan_schema"
PROGRAM_PERSISTENT_PLAN_DIGEST_KEY = "program_resource_plan_digest"
PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY = "program_resource_plan_maximum_bytes"
PROGRAM_PERSISTENT_SLOT_COUNT_KEY = "program_resource_slot_count"

PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA = "program-persistent-value-checkpoint:v1"
PROGRAM_PERSISTENT_PLAN_SCHEMA = "program-resource-plan:v1"
_PROGRAM_PERSISTENT_MAGIC = b"POPSPVS1"
_PROGRAM_PERSISTENT_TRAILER_BYTES = 8 + 64
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _capacity(value: Any, *, where: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an exact integer" % where)
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError("%s must be >= %d" % (where, minimum))
    if value > sys.maxsize:
        raise OverflowError("%s exceeds the native addressable range" % where)
    return value


@dataclass(frozen=True, slots=True)
class CheckpointResourceBudget:
    """Trusted allocation envelope installed from one authenticated live runtime."""

    runtime_kind: str
    max_members: int
    max_manifest_characters: int
    max_array_bytes: int
    max_uncompressed_bytes: int
    max_archive_bytes: int
    authority: str
    program_resource_plan_schema: str | None = None
    program_resource_plan_digest: str | None = None
    program_resource_plan_maximum_bytes: int | None = None
    program_resource_slot_count: int | None = None

    def __post_init__(self) -> None:
        if self.runtime_kind not in {"uniform", "amr", "multi_layout_uniform"}:
            raise ValueError("checkpoint resource budget has an unsupported runtime kind")
        for name in (
            "max_members",
            "max_manifest_characters",
            "max_array_bytes",
            "max_uncompressed_bytes",
            "max_archive_bytes",
        ):
            _capacity(getattr(self, name), where="checkpoint budget %s" % name, positive=True)
        if self.max_array_bytes > self.max_uncompressed_bytes:
            raise ValueError(
                "checkpoint per-array resource capacity exceeds its aggregate capacity"
            )
        if not isinstance(self.authority, str) or not self.authority:
            raise TypeError("checkpoint resource budget authority must be non-empty text")
        plan_fields = (
            self.program_resource_plan_schema,
            self.program_resource_plan_digest,
            self.program_resource_plan_maximum_bytes,
            self.program_resource_slot_count,
        )
        if any(value is not None for value in plan_fields):
            if (
                not isinstance(self.program_resource_plan_schema, str)
                or self.program_resource_plan_schema != PROGRAM_PERSISTENT_PLAN_SCHEMA
                or not isinstance(self.program_resource_plan_digest, str)
                or len(self.program_resource_plan_digest) != 64
                or any(char not in "0123456789abcdef" for char in self.program_resource_plan_digest)
            ):
                raise ValueError("checkpoint budget has an invalid Program resource-plan identity")
            _capacity(
                self.program_resource_plan_maximum_bytes,
                where="checkpoint Program resource-plan maximum_bytes",
            )
            _capacity(
                self.program_resource_slot_count,
                where="checkpoint Program resource-plan slot_count",
            )


def require_checkpoint_resource_budget(owner: Any) -> CheckpointResourceBudget:
    """Return the exact authenticated budget installed on one runtime owner."""
    budget = getattr(owner, "_checkpoint_resource_budget", None)
    if type(budget) is not CheckpointResourceBudget:
        raise RuntimeError("checkpoint decode requires the authenticated live resource budget")
    return budget


@dataclass(frozen=True, slots=True)
class ProgramPersistentValueCheckpoint:
    """Python projection of the native ``POPSPVS1`` checkpoint carrier.

    The C++ header is the normative codec.  Keeping this small, lossless projection in the
    Python checkpoint layer lets the NPZ manifest authenticate the opaque byte member and lets
    restart reject an old/incomplete image *before* asking the native engine to prepare anything.
    It intentionally exposes dictionaries rather than a second resource-plan implementation; the
    native detached prepare remains authoritative for target-plan equality and providers.
    """

    bound: bool
    schema: str
    plan_schema: str
    plan_digest: str
    maximum_bytes: int
    slot_count: int
    rows: tuple[dict[str, Any], ...]
    metadata: tuple[dict[str, Any], ...]
    offsets: tuple[int, ...]
    value_bytes: tuple[int, ...]
    storage: bytes

    def to_data(self) -> dict[str, Any]:
        return {
            "bound": self.bound,
            "schema": self.schema,
            "plan_schema": self.plan_schema,
            "plan_digest": self.plan_digest,
            "maximum_bytes": self.maximum_bytes,
            "slot_count": self.slot_count,
            "rows": [dict(row) for row in self.rows],
            "metadata": [dict(row) for row in self.metadata],
            "offsets": list(self.offsets),
            "value_bytes": list(self.value_bytes),
            "storage_bytes": len(self.storage),
        }


def _persistent_fail(reason: str) -> None:
    raise ValueError("invalid Program persistent value checkpoint: %s" % reason)


def _persistent_u8(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFF:
        _persistent_fail("%s is not a uint8" % where)
    return value


def _persistent_u32(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT32_MAX:
        _persistent_fail("%s is not a uint32" % where)
    return value


def _persistent_u64(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX:
        _persistent_fail("%s is not a uint64" % where)
    return value


def _persistent_i32(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not -(1 << 31) <= value < (1 << 31):
        _persistent_fail("%s is not an int32" % where)
    return value


def _persistent_text(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        _persistent_fail("%s is not text" % where)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _persistent_fail("%s is not valid UTF-8" % where)
    return value


class _PersistentReader:
    __slots__ = ("_data", "_offset")

    def __init__(self, payload: bytes) -> None:
        self._data = memoryview(payload)
        self._offset = 0

    def _read(self, size: int) -> bytes:
        if size < 0 or size > len(self._data) - self._offset:
            _persistent_fail("truncated payload")
        begin = self._offset
        self._offset += size
        return self._data[begin : begin + size].tobytes()

    def u8(self) -> int:
        return self._read(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def i32(self) -> int:
        raw = self.u64()
        value = raw - (1 << 64) if raw >= (1 << 63) else raw
        if not -(1 << 31) <= value < (1 << 31):
            _persistent_fail("signed integer is outside int32 range")
        return value

    def real(self) -> float:
        return struct.unpack("<d", self._read(8))[0]

    def string(self) -> str:
        size = self.u64()
        if size > len(self._data) - self._offset:
            _persistent_fail("string length is not credible")
        try:
            return self._read(size).decode("utf-8")
        except UnicodeDecodeError:
            _persistent_fail("string is not valid UTF-8")
        raise AssertionError("unreachable")

    def bytes(self) -> bytes:
        size = self.u64()
        if size > len(self._data) - self._offset:
            _persistent_fail("byte length is not credible")
        return self._read(size)

    def count(self, element_bytes: int = 1) -> int:
        size = self.u64()
        if element_bytes <= 0 or size > (len(self._data) - self._offset) // element_bytes:
            _persistent_fail("container length is not credible")
        return size

    def finish(self) -> None:
        if self._offset != len(self._data):
            _persistent_fail("trailing bytes")


def _persistent_read_optional_u64(reader: _PersistentReader) -> int | None:
    present = reader.u8()
    if present > 1:
        _persistent_fail("optional integer marker is invalid")
    return reader.u64() if present else None


def _persistent_read_row_wire(reader: _PersistentReader) -> dict[str, Any]:
    slot = reader.u32()
    value_id = reader.u64()
    occurrence_path_id = reader.u64()
    owner = reader.u32()
    space = reader.u32()
    clock = reader.u32()
    level = reader.i32()
    row = {
        "slot": slot,
        "key": {
            "value_id": value_id,
            "occurrence_path_id": occurrence_path_id,
            "owner": owner,
            "space": space,
            "clock": clock,
            "level": None if level == -1 else level,
        },
        "identity": reader.string(),
        "occurrence_path": reader.string(),
        "owner_identity": reader.string(),
        "space_identity": reader.string(),
        "clock_identity": reader.string(),
        "lifetime": reader.u8(),
        "centering": reader.u8(),
        "off_policy": reader.u8(),
        "spatial_transfer": reader.u8(),
        "components": reader.u32(),
        "ghosts": reader.u32(),
        "bytes": reader.u64(),
        "maximum_bytes": reader.u64(),
    }
    communicates = reader.u8()
    restart_required = reader.u8()
    if communicates > 1 or restart_required > 1:
        _persistent_fail("resource boolean is not 0 or 1")
    row["communicates"] = bool(communicates)
    row["restart_required"] = bool(restart_required)
    row["communication"] = reader.string()
    row["transfer_identity"] = reader.string()
    row["restart_identity"] = reader.string()
    row["component_names"] = reader.string()
    row["shape"] = reader.string()
    row["cells"] = _persistent_read_optional_u64(reader)
    row["itemsize"] = _persistent_read_optional_u64(reader)
    return row


def _persistent_read_metadata(reader: _PersistentReader) -> dict[str, Any]:
    accepted_coordinate = reader.u64()
    cursor = reader.u64()
    accumulated_dt = reader.real()
    topology_epoch = reader.u64()
    layout_generation = reader.u64()
    valid_value = reader.u8()
    cold_value = reader.u8()
    if valid_value > 1 or cold_value > 1:
        _persistent_fail("slot boolean is not 0 or 1")
    return {
        "accepted_coordinate": accepted_coordinate,
        "cursor": cursor,
        "accumulated_dt": accumulated_dt,
        "topology_epoch": topology_epoch,
        "layout_generation": layout_generation,
        "valid": bool(valid_value),
        "cold": bool(cold_value),
    }


def _persistent_validate_row(row: Mapping[str, Any], *, slot: int) -> None:
    if not isinstance(row, Mapping):
        _persistent_fail("resource row is not a mapping")
    key = row.get("key")
    if not isinstance(key, Mapping):
        _persistent_fail("resource row has no complete key")
    for name in ("value_id", "occurrence_path_id", "owner", "space", "clock"):
        _persistent_u64(key[name], where="resource key %s" % name) if name in {
            "value_id", "occurrence_path_id"
        } else _persistent_u32(key[name], where="resource key %s" % name)
    level = key.get("level")
    if level is not None:
        _persistent_u32(level, where="resource key level")
    if row.get("slot") != slot:
        _persistent_fail("resource rows are not dense")
    for name in ("identity", "occurrence_path", "owner_identity", "space_identity", "clock_identity",
                 "communication", "component_names", "shape"):
        if not _persistent_text(row.get(name), where="resource %s" % name):
            _persistent_fail("resource %s is empty" % name)
    for name in ("transfer_identity", "restart_identity"):
        _persistent_text(row.get(name), where="resource %s" % name)
    for name in ("lifetime", "centering", "off_policy", "spatial_transfer"):
        _persistent_u8(row.get(name), where="resource %s" % name)
    if row["lifetime"] not in (1, 2):
        _persistent_fail("resource row has an unknown lifetime")
    if row["centering"] not in (1, 2, 3):
        _persistent_fail("resource row has an unknown centering")
    if row["off_policy"] not in (0, 1, 2, 3, 4):
        _persistent_fail("resource row has an unknown off-schedule policy")
    if row["spatial_transfer"] not in (1, 2, 3):
        _persistent_fail("resource row has an unknown transfer policy")
    components = _persistent_u32(row.get("components"), where="resource components")
    ghosts = _persistent_u32(row.get("ghosts"), where="resource ghosts")
    del ghosts
    value_bytes = _persistent_u64(row.get("bytes"), where="resource bytes")
    maximum_bytes = _persistent_u64(row.get("maximum_bytes"), where="resource maximum_bytes")
    if components == 0 or value_bytes == 0 or maximum_bytes < value_bytes:
        _persistent_fail("resource row has an incomplete byte contract")
    for name in ("communicates", "restart_required"):
        if type(row.get(name)) is not bool:
            _persistent_fail("resource %s is not a bool" % name)
    if row["lifetime"] == 1 and row["off_policy"] != 0:
        _persistent_fail("transient resource carries an off-schedule policy")
    if row["spatial_transfer"] == 2 and not row["transfer_identity"]:
        _persistent_fail("qualified regrid resource has no provider")
    if row["restart_required"] and not row["restart_identity"]:
        _persistent_fail("restart-required resource has no provider")
    for name in ("cells", "itemsize"):
        value = row.get(name)
        if value is not None and _persistent_u64(value, where="resource %s" % name) == 0:
            _persistent_fail("resource optional extent is zero")


def validate_program_persistent_value_checkpoint(
    image: ProgramPersistentValueCheckpoint,
) -> ProgramPersistentValueCheckpoint:
    """Validate a decoded image with the same lossless guards as the native v1 codec."""
    if type(image) is not ProgramPersistentValueCheckpoint:
        raise TypeError("Program persistent checkpoint requires its exact image type")
    if type(image.bound) is not bool or image.schema != PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA:
        _persistent_fail("unsupported checkpoint schema")
    if not image.bound:
        if any((image.plan_schema, image.plan_digest, image.maximum_bytes, image.slot_count,
                image.rows, image.metadata, image.offsets, image.value_bytes, image.storage)):
            _persistent_fail("unbound checkpoint owns a resource image")
        return image
    if image.plan_schema != PROGRAM_PERSISTENT_PLAN_SCHEMA:
        _persistent_fail("unsupported resource-plan schema")
    if (not isinstance(image.plan_digest, str) or len(image.plan_digest) != 64
            or any(char not in "0123456789abcdef" for char in image.plan_digest)):
        _persistent_fail("resource-plan digest is not lowercase SHA-256")
    if _persistent_u32(image.slot_count, where="checkpoint slot_count") != len(image.rows):
        _persistent_fail("checkpoint slot count disagrees with rows")
    if len(image.rows) != len(image.metadata) or len(image.rows) != len(image.value_bytes):
        _persistent_fail("row, metadata and value-size counts disagree")
    if len(image.offsets) != len(image.rows) + 1 or not image.offsets:
        _persistent_fail("checkpoint offset count disagrees with rows")
    if not isinstance(image.storage, (bytes, bytearray, memoryview)):
        _persistent_fail("storage image is not bytes")
    storage_size = len(image.storage)
    maximum = _persistent_u64(image.maximum_bytes, where="checkpoint maximum_bytes")
    if image.offsets[0] != 0 or image.offsets[-1] != storage_size:
        _persistent_fail("storage offsets do not cover the exact storage image")
    previous = 0
    total = 0
    complete_keys: dict[tuple[Any, ...], int] = {}
    path_digests: dict[int, str] = {}
    identities: dict[str, int] = {}
    for slot, row in enumerate(image.rows):
        offset = _persistent_u64(image.offsets[slot], where="checkpoint offset")
        end = _persistent_u64(image.offsets[slot + 1], where="checkpoint offset")
        if offset < previous or end < offset or end > maximum:
            _persistent_fail("storage offsets exceed the authenticated memory bound")
        previous = end
        _persistent_validate_row(row, slot=slot)
        key = row["key"]
        complete = (
            key["value_id"], key.get("level"), row["occurrence_path"],
            row["owner_identity"], row["space_identity"], row["clock_identity"],
        )
        if complete in complete_keys:
            _persistent_fail("resource plan has a duplicate complete key")
        complete_keys[complete] = slot
        occurrence_path_id = key["occurrence_path_id"]
        prior_path = path_digests.get(occurrence_path_id)
        if prior_path is not None and prior_path != row["occurrence_path"]:
            _persistent_fail("resource plan occurrence digest collision")
        path_digests[occurrence_path_id] = row["occurrence_path"]
        prior_identity = identities.get(row["identity"])
        if prior_identity is not None and prior_identity != slot:
            _persistent_fail("resource plan has a duplicate identity")
        identities[row["identity"]] = slot
        extent = end - offset
        value_size = _persistent_u64(image.value_bytes[slot], where="checkpoint value_bytes")
        metadata = image.metadata[slot]
        if not isinstance(metadata, Mapping):
            _persistent_fail("slot metadata is not a mapping")
        for name in ("accepted_coordinate", "cursor", "topology_epoch", "layout_generation"):
            _persistent_u64(metadata.get(name), where="slot %s" % name)
        for name in ("valid", "cold"):
            if type(metadata.get(name)) is not bool:
                _persistent_fail("slot %s is not a bool" % name)
        accumulated_dt = metadata.get("accumulated_dt")
        if isinstance(accumulated_dt, bool) or not isinstance(accumulated_dt, (int, float)) \
                or not math.isfinite(accumulated_dt) or accumulated_dt < 0.0:
            _persistent_fail("slot accumulated_dt is not finite and non-negative")
        if metadata["valid"] == metadata["cold"]:
            _persistent_fail("slot validity and cold markers are not complementary")
        expected_value_size = row["bytes"] if metadata["valid"] else 0
        if value_size != expected_value_size or extent != row["maximum_bytes"] or value_size > extent:
            _persistent_fail("dense row and storage metadata disagree")
        if total > _UINT64_MAX - row["maximum_bytes"]:
            raise OverflowError("Program persistent checkpoint memory bound overflows uint64")
        total += row["maximum_bytes"]
    # The dense carrier owns every byte reserved by its sealed resource plan.  Treating the
    # header as merely an upper bound would admit a truncated/incomplete plan with a larger
    # resigned ceiling, particularly on rank-redistribution and regrid routes where the source
    # and target digests are intentionally allowed to differ.
    if total != maximum or image.offsets[-1] != maximum:
        _persistent_fail("resource rows do not cover the exact checkpoint memory ceiling")
    return image


def _persistent_decode_body(body: bytes) -> ProgramPersistentValueCheckpoint:
    reader = _PersistentReader(body)
    if reader._read(len(_PROGRAM_PERSISTENT_MAGIC)) != _PROGRAM_PERSISTENT_MAGIC:
        _persistent_fail("unsupported checkpoint magic")
    bound = reader.u8()
    if bound > 1:
        _persistent_fail("bound marker is invalid")
    image = {
        "bound": bool(bound),
        "schema": reader.string(),
        "plan_schema": reader.string(),
        "plan_digest": reader.string(),
        "maximum_bytes": reader.u64(),
        "slot_count": reader.u32(),
    }
    rows_count = reader.count()
    image["rows"] = tuple(_persistent_read_row_wire(reader) for _ in range(rows_count))
    metadata_count = reader.count()
    image["metadata"] = tuple(_persistent_read_metadata(reader) for _ in range(metadata_count))
    offsets_count = reader.count(8)
    image["offsets"] = tuple(reader.u64() for _ in range(offsets_count))
    value_count = reader.count(8)
    image["value_bytes"] = tuple(reader.u64() for _ in range(value_count))
    image["storage"] = reader.bytes()
    reader.finish()
    return ProgramPersistentValueCheckpoint(**image)


def decode_program_persistent_value_checkpoint(payload: Any) -> ProgramPersistentValueCheckpoint:
    """Decode/authenticate one native ``POPSPVS1`` byte image without touching live state."""
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    elif isinstance(payload, bytearray):
        payload = bytes(payload)
    if type(payload) is not bytes:
        raise TypeError("Program persistent checkpoint must be an exact bytes image")
    if len(payload) < _PROGRAM_PERSISTENT_TRAILER_BYTES:
        _persistent_fail("truncated checksum trailer")
    body = payload[:-_PROGRAM_PERSISTENT_TRAILER_BYTES]
    trailer = payload[-_PROGRAM_PERSISTENT_TRAILER_BYTES:]
    expected_size = struct.unpack("<Q", trailer[:8])[0]
    if expected_size != 64 or trailer[8:].decode("ascii", errors="replace") != hashlib.sha256(body).hexdigest():
        _persistent_fail("checkpoint SHA-256 digest mismatch")
    image = _persistent_decode_body(body)
    return validate_program_persistent_value_checkpoint(image)


def program_persistent_checkpoint_from_payload(
    payload: Mapping[str, Any],
) -> tuple[bytes, ProgramPersistentValueCheckpoint]:
    """Decode one NPZ carrier and authenticate its redundant scalar envelope members.

    The scalar members are not a second source of truth: they are a bounded, inspectable copy of
    the native schema/plan identity.  Requiring all six and comparing them before native prepare
    makes an old or partially copied archive fail deterministically without touching live state.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Program persistent checkpoint payload must be a mapping")
    raw_value = payload.get(PROGRAM_PERSISTENT_CHECKPOINT_KEY)
    if raw_value is None:
        raise ValueError("checkpoint lacks the ProgramPersistentValueCheckpoint carrier")
    if isinstance(raw_value, memoryview):
        raw = raw_value.tobytes()
    elif isinstance(raw_value, bytearray):
        raw = bytes(raw_value)
    elif type(raw_value) is bytes:
        raw = raw_value
    else:
        import numpy as np

        array = np.asarray(raw_value)
        if array.dtype != np.dtype(np.uint8) or array.ndim != 1 or not array.flags.c_contiguous:
            raise TypeError(
                "Program persistent checkpoint NPZ member must be a C-contiguous uint8 vector"
            )
        raw = array.tobytes(order="C")
    image = decode_program_persistent_value_checkpoint(raw)

    import numpy as np

    def scalar(name: str) -> Any:
        if name not in payload:
            raise ValueError("checkpoint lacks Program persistent member %r" % name)
        value = np.asarray(payload[name])
        if value.shape != ():
            raise ValueError("checkpoint Program persistent member %r is not scalar" % name)
        return value.item()

    schema = scalar(PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY)
    plan_schema = scalar(PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY)
    plan_digest = scalar(PROGRAM_PERSISTENT_PLAN_DIGEST_KEY)
    maximum_bytes = scalar(PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY)
    slot_count = scalar(PROGRAM_PERSISTENT_SLOT_COUNT_KEY)
    if not isinstance(schema, str) or schema != image.schema:
        raise ValueError("checkpoint Program persistent schema member differs from its carrier")
    if not isinstance(plan_schema, str) or plan_schema != image.plan_schema:
        raise ValueError("checkpoint Program persistent plan schema differs from its carrier")
    if not isinstance(plan_digest, str) or plan_digest != image.plan_digest:
        raise ValueError("checkpoint Program persistent plan digest differs from its carrier")
    if isinstance(maximum_bytes, (bool, np.bool_)) or not isinstance(maximum_bytes, (int, np.integer)):
        raise TypeError("checkpoint Program persistent maximum_bytes must be an integer scalar")
    if isinstance(slot_count, (bool, np.bool_)) or not isinstance(slot_count, (int, np.integer)):
        raise TypeError("checkpoint Program persistent slot_count must be an integer scalar")
    if int(maximum_bytes) != image.maximum_bytes:
        raise ValueError("checkpoint Program persistent maximum_bytes differs from its carrier")
    if int(slot_count) != image.slot_count:
        raise ValueError("checkpoint Program persistent slot_count differs from its carrier")
    return raw, image


def require_program_persistent_checkpoint_plan(
    image: ProgramPersistentValueCheckpoint,
    budget: CheckpointResourceBudget,
    *,
    mode: str = "restore_recorded_hierarchy",
) -> ProgramPersistentValueCheckpoint:
    """Authenticate the source carrier against the live target-plan authority.

    An unchanged hierarchy must match the live plan byte-for-byte.  Rank redistribution and
    qualified regrid deliberately permit a different runtime-sized target footprint; their
    dedicated native preparation seams compare the complete source rows and transfer policies to
    the complete target plan.  Applying the same-layout digest equality first would make those
    specialized seams unreachable.
    """
    validate_program_persistent_value_checkpoint(image)
    if type(budget) is not CheckpointResourceBudget:
        raise TypeError("Program persistent checkpoint requires an exact live resource budget")
    if not image.bound:
        raise ValueError("checkpoint Program persistent value image is unbound")
    if type(mode) is not str or mode not in {
        "restore_recorded_hierarchy",
        "rank_change",
        "regrid_on_restart",
    }:
        raise ValueError("unknown Program persistent checkpoint restore mode %r" % mode)
    expected = (
        budget.program_resource_plan_schema,
        budget.program_resource_plan_digest,
        budget.program_resource_plan_maximum_bytes,
        budget.program_resource_slot_count,
    )
    if any(value is None for value in expected):
        raise RuntimeError("live checkpoint budget lacks the sealed Program resource-plan identity")
    if mode == "restore_recorded_hierarchy" and expected != (
        image.plan_schema,
        image.plan_digest,
        image.maximum_bytes,
        image.slot_count,
    ):
        raise ValueError("checkpoint Program resource plan differs from the bound live plan")
    return image


class _PersistentWriter:
    __slots__ = ("parts",)

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def raw(self, value: bytes) -> None:
        self.parts.append(value)

    def u8(self, value: int) -> None:
        self.parts.append(struct.pack("<B", value))

    def u32(self, value: int) -> None:
        self.parts.append(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        self.parts.append(struct.pack("<Q", value))

    def i32(self, value: int) -> None:
        self.u64(value if value >= 0 else (1 << 64) + value)

    def real(self, value: float) -> None:
        self.parts.append(struct.pack("<d", value))

    def string(self, value: str) -> None:
        raw = value.encode("utf-8")
        self.u64(len(raw))
        self.raw(raw)

    def bytes(self, value: bytes) -> None:
        self.u64(len(value))
        self.raw(value)

    def take(self) -> bytes:
        return b"".join(self.parts)


def _persistent_write_optional_u64(writer: _PersistentWriter, value: int | None) -> None:
    writer.u8(0 if value is None else 1)
    if value is not None:
        writer.u64(value)


def _persistent_row_data(row: Mapping[str, Any]) -> dict[str, Any]:
    key = row.get("key")
    if not isinstance(key, Mapping):
        _persistent_fail("resource row has no complete key")
    level = key.get("level", key.get("amr_level"))
    return {
        "slot": row.get("slot"),
        "key": {
            "value_id": key.get("value_id"),
            "occurrence_path_id": key.get("occurrence_path_id"),
            "owner": key.get("owner"),
            "space": key.get("space"),
            "clock": key.get("clock"),
            "level": level,
        },
        **{name: row.get(name) for name in (
            "identity", "occurrence_path", "owner_identity", "space_identity", "clock_identity",
            "lifetime", "centering", "off_policy", "spatial_transfer", "components", "ghosts",
            "bytes", "maximum_bytes", "communicates", "restart_required", "communication",
            "transfer_identity", "restart_identity", "component_names", "shape", "cells", "itemsize",
        )},
    }


def _persistent_encode_image(image: ProgramPersistentValueCheckpoint) -> bytes:
    validate_program_persistent_value_checkpoint(image)
    writer = _PersistentWriter()
    writer.raw(_PROGRAM_PERSISTENT_MAGIC)
    writer.u8(1 if image.bound else 0)
    writer.string(image.schema)
    writer.string(image.plan_schema)
    writer.string(image.plan_digest)
    writer.u64(image.maximum_bytes)
    writer.u32(image.slot_count)
    writer.u64(len(image.rows))
    for source in image.rows:
        row = _persistent_row_data(source)
        key = row["key"]
        writer.u32(row["slot"])
        writer.u64(key["value_id"])
        writer.u64(key["occurrence_path_id"])
        writer.u32(key["owner"])
        writer.u32(key["space"])
        writer.u32(key["clock"])
        writer.i32(-1 if key["level"] is None else key["level"])
        for name in ("identity", "occurrence_path", "owner_identity", "space_identity", "clock_identity"):
            writer.string(row[name])
        for name in ("lifetime", "centering", "off_policy", "spatial_transfer"):
            writer.u8(row[name])
        for name in ("components", "ghosts"):
            writer.u32(row[name])
        for name in ("bytes", "maximum_bytes"):
            writer.u64(row[name])
        writer.u8(1 if row["communicates"] else 0)
        writer.u8(1 if row["restart_required"] else 0)
        for name in ("communication", "transfer_identity", "restart_identity", "component_names", "shape"):
            writer.string(row[name])
        _persistent_write_optional_u64(writer, row["cells"])
        _persistent_write_optional_u64(writer, row["itemsize"])
    writer.u64(len(image.metadata))
    for metadata in image.metadata:
        for name in ("accepted_coordinate", "cursor"):
            writer.u64(metadata[name])
        writer.real(metadata["accumulated_dt"])
        for name in ("topology_epoch", "layout_generation"):
            writer.u64(metadata[name])
        writer.u8(1 if metadata["valid"] else 0)
        writer.u8(1 if metadata["cold"] else 0)
    writer.u64(len(image.offsets))
    for value in image.offsets:
        writer.u64(value)
    writer.u64(len(image.value_bytes))
    for value in image.value_bytes:
        writer.u64(value)
    writer.bytes(image.storage)
    body = writer.take()
    trailer = _PersistentWriter()
    trailer.string(hashlib.sha256(body).hexdigest())
    return body + trailer.take()


def encode_program_persistent_value_checkpoint(image: Any) -> bytes:
    """Encode a test/host image in the exact native v1 wire format."""
    if type(image) is not ProgramPersistentValueCheckpoint:
        raise TypeError("Program persistent checkpoint requires its exact image type")
    return _persistent_encode_image(image)


def program_persistent_checkpoint_manifest(image: ProgramPersistentValueCheckpoint) -> dict[str, Any]:
    """Return the exact compact manifest extension embedded in checkpoint envelopes."""
    validate_program_persistent_value_checkpoint(image)
    return {
        "schema": image.schema,
        "plan_schema": image.plan_schema,
        "plan_digest": image.plan_digest,
        "maximum_bytes": image.maximum_bytes,
        "slot_count": image.slot_count,
    }


def program_persistent_checkpoint_manifest_from_payload(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = payload.get(PROGRAM_PERSISTENT_CHECKPOINT_KEY)
    if raw is None:
        return None
    _raw, image = program_persistent_checkpoint_from_payload(payload)
    return program_persistent_checkpoint_manifest(image)


def _persistent_plan_row_strings(row: Any) -> tuple[str, ...]:
    names = row.component_names
    if not isinstance(names, str):
        names = json.dumps(list(names), ensure_ascii=True, separators=(",", ":"))
    shape = row.shape
    if not isinstance(shape, str):
        shape = json.dumps(list(shape), ensure_ascii=True, separators=(",", ":"))
    return (
        row.identity,
        row.key.occurrence_path,
        row.key.owner,
        row.key.space,
        row.key.clock,
        row.communication,
        row.transfer_provider,
        row.restart_provider,
        names,
        shape,
    )


def program_persistent_value_checkpoint_capacity(plan: Any) -> int:
    """Return an exact upper bound for the native image serialized from one sealed plan.

    ``ProgramPersistentValueStore`` allocates ``sum(row.maximum_bytes)`` eagerly.  The returned
    bound therefore includes that exact storage ceiling plus every fixed/variable metadata field,
    offsets, value sizes and the SHA-256 trailer.  It is used at bind time, never inferred from a
    live cache or from a zero-byte placeholder.
    """
    if not isinstance(plan, ProgramResourcePlanCapacityAuthority):
        raise TypeError(
            "Program persistent checkpoint capacity requires an exact sealed ProgramResourcePlan"
        )
    rows = plan.abi_rows()
    storage = 0
    size = 8 + 1  # magic + bound marker
    size += 8 + len(PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA.encode())
    size += 8 + len(PROGRAM_PERSISTENT_PLAN_SCHEMA.encode())
    digest = str(getattr(plan, "digest", ""))
    size += 8 + len(digest.encode())
    size += 8 + 4  # maximum_bytes + slot_count
    size += 8  # row count
    for row in rows:
        storage += int(row.maximum_bytes)
        size += 4 + 8 + 8 + 4 + 4 + 4 + 8  # slot/key ids/level
        size += sum(8 + len(value.encode("utf-8")) for value in _persistent_plan_row_strings(row)[:5])
        size += 4  # typed policy bytes
        size += 4 + 4 + 8 + 8  # components, ghosts, bytes, maximum_bytes
        size += 2  # communicates/restart_required
        size += sum(8 + len(value.encode("utf-8")) for value in _persistent_plan_row_strings(row)[5:])
        size += (1 + 8 if row.cells is not None else 1)
        size += (1 + 8 if row.itemsize is not None else 1)
    size += 8 + len(rows) * 42  # metadata count + fixed metadata rows
    size += 8 + (len(rows) + 1) * 8  # offsets
    size += 8 + len(rows) * 8  # logical value sizes
    size += 8 + storage  # storage byte vector
    size += _PROGRAM_PERSISTENT_TRAILER_BYTES
    return _capacity(size, where="Program persistent checkpoint byte capacity", positive=True)


def _persistent_native_method(native: Any, names: tuple[str, ...]) -> tuple[str, Callable[..., Any]] | None:
    for name in names:
        method = getattr(native, name, None)
        if callable(method):
            return name, method
    return None


def _persistent_native_has_program(native: Any) -> bool:
    provider = getattr(native, "installed_program_hash", None)
    if callable(provider):
        return bool(str(provider()))
    value = getattr(native, "installed_program_hash", "")
    return isinstance(value, str) and bool(value)


def program_persistent_value_checkpoint_capture_available(native: Any) -> bool:
    """Return whether the native host exposes the POPSPVS1 capture carrier.

    This is an introspection-only capability query.  It deliberately does not inspect the live
    store or invoke a collective callback.  The exact method name is part of the POPSPVS1 binding
    contract; legacy cache serializers are not accepted as a fallback.
    """

    return callable(getattr(native, "capture_program_persistent_value_checkpoint", None))


def require_program_persistent_value_checkpoint_capture(native: Any) -> None:
    """Reject a compiled Program before capture if its lossless host seam is unavailable.

    This probe intentionally does not call the native method: a binding is allowed to implement
    capture as a collective operation, and preparation must remain side-effect free until the
    collective capture phase.  A native host without the exact seam is rejected for a compiled
    Program; no legacy cache serializer can satisfy the contract.
    """
    if not program_persistent_value_checkpoint_capture_available(native) and _persistent_native_has_program(native):
        raise RuntimeError(
            "checkpoint compiled Program lacks the ProgramPersistentValueCheckpoint capture seam"
        )


def capture_program_persistent_value_checkpoint(native: Any) -> tuple[bytes, ProgramPersistentValueCheckpoint] | None:
    """Capture one native image through the versioned host seam, or refuse a compiled gap."""
    method = _persistent_native_method(native, ("capture_program_persistent_value_checkpoint",))
    if method is None:
        if _persistent_native_has_program(native):
            raise RuntimeError(
                "checkpoint compiled Program lacks the ProgramPersistentValueCheckpoint capture seam"
            )
        return None
    raw = method[1]()
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    if type(raw) is not bytes:
        raise TypeError("native Program persistent checkpoint capture must return bytes")
    return raw, decode_program_persistent_value_checkpoint(raw)


@dataclass(frozen=True, slots=True)
class PreparedProgramPersistentValueRestore:
    """Detached native restore plus its non-throwing publication callback."""

    image: ProgramPersistentValueCheckpoint
    native_image: Any
    publish: Callable[[Any], Any]


def prepare_program_persistent_value_restore(
    native: Any,
    payload: Any,
    *,
    mode: str = "restore_recorded_hierarchy",
) -> PreparedProgramPersistentValueRestore:
    """Decode and prepare a persistent image without mutating the live native store.

    Rank-change and qualified-regrid paths intentionally require dedicated native preparation
    seams.  Falling back to a same-layout restore would silently discard transfer/provider policy.
    """
    image = decode_program_persistent_value_checkpoint(payload)
    if type(mode) is not str or mode not in {
        "restore_recorded_hierarchy",
        "rank_change",
        "regrid_on_restart",
    }:
        raise ValueError("unknown Program persistent checkpoint restore mode %r" % mode)
    if mode == "rank_change":
        names = ("prepare_program_persistent_value_redistribution",)
    elif mode == "regrid_on_restart":
        names = ("prepare_program_persistent_value_regrid",)
    else:
        names = ("prepare_program_persistent_value_restore",)
    method = _persistent_native_method(native, names)
    if method is None:
        raise RuntimeError(
            "native Program lacks the detached ProgramPersistentValueCheckpoint %s seam" % mode
        )
    try:
        native_image = method[1](payload)
    except TypeError as error:
        # New bindings may expose the mode as an optional keyword while older prepared bindings
        # use a bytes-only method.  Do not retry a specialized method: its argument contract is
        # intentionally exact and a TypeError from validation must remain visible.
        if method[0] not in {
            "prepare_program_persistent_value_restore",
        }:
            raise
        try:
            native_image = method[1](payload, mode=mode)
        except TypeError:
            raise error
    if native_image is None:
        raise RuntimeError("native Program persistent checkpoint prepare returned no detached image")
    publish_method = _persistent_native_method(
        native, ("publish_program_persistent_value_restore",)
    )
    if publish_method is None:
        raise RuntimeError(
            "native Program lacks the non-throwing ProgramPersistentValueCheckpoint publish seam"
        )
    return PreparedProgramPersistentValueRestore(image, native_image, publish_method[1])


def publish_program_persistent_value_restore(prepared: PreparedProgramPersistentValueRestore) -> Any:
    if type(prepared) is not PreparedProgramPersistentValueRestore:
        raise TypeError("Program persistent restore requires its exact prepared image")
    return prepared.publish(prepared.native_image)


__all__ = [
    "CheckpointResourceBudget",
    "IDENTITY_KEY",
    "MANIFEST_KEY",
    "PROGRAM_PERSISTENT_CHECKPOINT_KEY",
    "PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA",
    "PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY",
    "PROGRAM_PERSISTENT_PLAN_DIGEST_KEY",
    "PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY",
    "PROGRAM_PERSISTENT_PLAN_SCHEMA",
    "PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY",
    "PROGRAM_PERSISTENT_SLOT_COUNT_KEY",
    "PreparedProgramPersistentValueRestore",
    "ProgramPersistentValueCheckpoint",
    "capture_program_persistent_value_checkpoint",
    "decode_program_persistent_value_checkpoint",
    "encode_program_persistent_value_checkpoint",
    "prepare_program_persistent_value_restore",
    "program_persistent_checkpoint_manifest",
    "program_persistent_checkpoint_manifest_from_payload",
    "program_persistent_checkpoint_from_payload",
    "program_persistent_value_checkpoint_capture_available",
    "require_program_persistent_checkpoint_plan",
    "program_persistent_value_checkpoint_capacity",
    "publish_program_persistent_value_restore",
    "require_program_persistent_value_checkpoint_capture",
    "require_checkpoint_resource_budget",
    "validate_program_persistent_value_checkpoint",
]
