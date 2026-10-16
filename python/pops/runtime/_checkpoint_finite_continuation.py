"""Archive transport for the one authoritative native structural continuation image.

Python authenticates archive shape/rank selection and forwards bytes; all type, alias, field,
frontier and capacity checks are native. No Python dictionary is a continuation runtime.
"""
from __future__ import annotations

import numpy as np

FINITE_CONTINUATION_KEYS = frozenset(("finite_continuation_schema", "finite_continuation_bytes",
                                     "finite_continuation_offsets"))
_SCHEMA = "finite_continuation_schema"
_BYTES = "finite_continuation_bytes"
_OFFSETS = "finite_continuation_offsets"


def installed_capacity(owner):
    schema = owner._s._checkpoint_continuation_schema()
    capacity = owner._s._checkpoint_continuation_capacity()
    if type(schema) is not str or type(capacity) is not int or capacity < 0 \
            or bool(schema) != bool(capacity):
        raise RuntimeError("native continuation schema/capacity authority is inconsistent")
    return schema, capacity


def capture_finite_continuation(owner, payload):
    from pops.output._checkpoint_collective import checkpoint_topology, consensus
    from pops._native_collectives import allgather_bytes

    topology = checkpoint_topology(owner)
    error = None
    local = b""
    schema = ""
    capacity = 0
    try:
        schema, capacity = installed_capacity(owner)
    except BaseException as exc:
        error = exc
    rows = consensus(topology, "finite continuation installed schema", error=error,
                     value=(schema, capacity))
    if any(row["value"] != rows[0]["value"] for row in rows[1:]):
        raise RuntimeError("finite continuation installed schema differs between ranks")
    if not schema:
        return
    # Native capture owns its own allocation/fence/error rendezvous. Every rank enters it.
    error = None
    try:
        local = owner._s._checkpoint_continuation_image()
        if type(local) is not bytes or not local or len(local) > capacity:
            raise ValueError("native continuation image exceeds its installed byte capacity")
    except BaseException as exc:
        error = exc
    consensus(topology, "finite continuation local capture", error=error)
    images = allgather_bytes(topology.communicator, local) if topology.distributed else (local,)
    error = None
    try:
        if len(images) != topology.size or any(not image or len(image) > capacity for image in images):
            raise ValueError("finite continuation rank image is absent or exceeds capacity")
        offsets = [0]
        for image in images:
            offsets.append(offsets[-1] + len(image))
        if offsets[-1] > np.iinfo(np.int64).max:
            raise OverflowError("finite continuation archive offsets exceed Int64")
        payload[_SCHEMA] = np.asarray(schema)
        payload[_BYTES] = np.frombuffer(b"".join(images), dtype=np.uint8).copy()
        payload[_OFFSETS] = np.asarray(offsets, dtype=np.int64)
    except BaseException as exc:
        error = exc
    consensus(topology, "finite continuation ranked archive", error=error)


def prepare_finite_continuation(owner, payload):
    from pops.output._checkpoint_collective import checkpoint_topology

    schema, capacity = installed_capacity(owner)
    present = FINITE_CONTINUATION_KEYS.intersection(payload)
    if not schema:
        if present:
            raise ValueError("checkpoint continuation has no matching installed schema")
        return None
    if present != FINITE_CONTINUATION_KEYS:
        raise ValueError("checkpoint omits part of the required native continuation")
    name = np.asarray(payload[_SCHEMA])
    data = np.asarray(payload[_BYTES])
    offsets = np.asarray(payload[_OFFSETS])
    topology = checkpoint_topology(owner)
    if name.ndim != 0 or name.dtype.kind != "U" or str(name) != schema:
        raise ValueError("checkpoint continuation schema differs from installed code")
    if data.ndim != 1 or data.dtype != np.dtype(np.uint8) or offsets.ndim != 1 \
            or offsets.dtype != np.dtype(np.int64) or len(offsets) != topology.size + 1 \
            or offsets[0] != 0 or offsets[-1] != len(data) \
            or np.any(offsets[1:] <= offsets[:-1]) \
            or any(int(b) - int(a) > capacity for a, b in zip(offsets[:-1], offsets[1:], strict=True)):
        raise ValueError("checkpoint continuation rank mapping, dtype or bounded offsets differ")
    # The initial profile keeps the exact distribution/rank count. It never chooses rank zero's
    # field shard as a substitute for missing or differently partitioned shards.
    image = data[int(offsets[topology.rank]):int(offsets[topology.rank + 1])].tobytes()
    time = np.asarray(payload.get("t"))
    step = np.asarray(payload.get("macro_step"))
    if time.ndim != 0 or time.dtype.kind != "f" or not np.isfinite(time) \
            or step.ndim != 0 or step.dtype.kind not in "iu" or not 0 <= int(step) <= 2**31 - 1:
        raise ValueError("checkpoint continuation lacks an exact native root frontier")
    owner._s._validate_checkpoint_continuation_image(image, float(time), int(step))
    return image
