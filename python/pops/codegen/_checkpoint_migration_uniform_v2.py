"""Exact historical Uniform v2 checkpoint schema used only by offline migration."""
from __future__ import annotations

import math
from typing import Any


HISTORICAL_UNIFORM_PAYLOAD_VERSION = 2


def _scalar_int(payload: Any, key: str, *, minimum: int = 0) -> int:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "iu":
        raise TypeError("historical Uniform checkpoint %r must be an integer scalar" % key)
    result = int(value.item())
    if result < minimum:
        raise ValueError(
            "historical Uniform checkpoint %r must be >= %d" % (key, minimum))
    return result


def _scalar_float(payload: Any, key: str) -> float:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind != "f":
        raise TypeError("historical Uniform checkpoint %r must be a floating scalar" % key)
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("historical Uniform checkpoint %r must be finite" % key)
    return result


def _text_scalar(payload: Any, key: str) -> str:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "US":
        raise TypeError("historical Uniform checkpoint %r must be a text scalar" % key)
    result = str(value.item())
    if not result:
        raise ValueError("historical Uniform checkpoint %r must be non-empty" % key)
    return result


def _text_vector(payload: Any, key: str) -> tuple[str, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise TypeError("historical Uniform checkpoint %r must be a text vector" % key)
    result = tuple(str(item) for item in value.tolist())
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(
            "historical Uniform checkpoint %r must contain unique non-empty text" % key)
    return result


def _int_vector(payload: Any, key: str) -> tuple[int, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise TypeError("historical Uniform checkpoint %r must be an integer vector" % key)
    return tuple(int(item) for item in value.tolist())


def _historical_dense_policy(payload: Any, name: str, depth: int) -> tuple[Any, tuple[int, ...]]:
    from pops._manifest_protocol import strict_json_loads
    from pops.time._history.persistence import Dense, Interval, Revolve

    raw = strict_json_loads(
        _text_scalar(payload, "history_policy_" + name),
        where="historical history persistence JSON",
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise ValueError(
            "historical history %r has no exact v2 persistence manifest" % name)
    kind = raw["kind"]
    if kind == "dense" and set(raw) == {"kind"}:
        policy = Dense()
    elif kind == "interval" and set(raw) == {"kind", "k"} \
            and isinstance(raw["k"], int) and not isinstance(raw["k"], bool):
        policy = Interval(raw["k"])
    elif kind == "revolve" and set(raw) == {"kind", "snapshots"} \
            and isinstance(raw["snapshots"], int) \
            and not isinstance(raw["snapshots"], bool):
        policy = Revolve(raw["snapshots"])
    else:
        raise ValueError(
            "historical history %r has an unsupported v2 persistence manifest" % name)
    policy.validate_for(depth)
    stored = _int_vector(payload, "history_stored_slots_" + name)
    expected = tuple(policy.stored_slots(depth))
    if stored != expected:
        raise ValueError(
            "historical history %r stored slots differ from its v2 policy" % name)
    if stored != tuple(range(depth)):
        raise ValueError(
            "offline Uniform v2 migration supports only fully stored history rings; "
            "selective replay metadata is not derivable for history %r" % name)
    return policy, stored


def validate_historical_v2(payload: Any) -> dict[str, Any]:
    """Validate the exact supported v2 array schema and return its detached facts."""
    import numpy as np
    from pops.runtime._checkpoint_manifest import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        require_exact_payload_version,
    )

    require_exact_payload_version(
        payload,
        key="pops_checkpoint_version",
        expected=HISTORICAL_UNIFORM_PAYLOAD_VERSION,
        runtime_kind="historical Uniform",
    )
    time = _scalar_float(payload, "t")
    macro_step = _scalar_int(payload, "macro_step")
    nx = _scalar_int(payload, "nx", minimum=1)
    ny = _scalar_int(payload, "ny", minimum=1)
    _text_scalar(payload, "abi_key")
    program_hash = _text_scalar(payload, "program_hash")
    if len(program_hash) != 64:
        raise ValueError("historical compiled Program hash must contain 64 hexadecimal digits")
    try:
        bytes.fromhex(program_hash)
    except ValueError:
        raise ValueError("historical compiled Program hash is not hexadecimal") from None

    required = {
        "pops_checkpoint_version", "t", "macro_step", "nx", "ny", "abi_key",
        "blocks", "phi", "program_hash", "history_names", "cache_nodes", "cache_names",
    }
    blocks = _text_vector(payload, "blocks")
    if not blocks:
        raise ValueError("historical Uniform checkpoint must contain at least one block")
    block_rows = []
    for block in blocks:
        keys = {"ncomp_" + block, "state_" + block, "names_" + block}
        required.update(keys)
        if not keys <= set(payload.files):
            raise ValueError(
                "historical Uniform block %r has an incomplete v2 payload" % block)
        ncomp = _scalar_int(payload, "ncomp_" + block, minimum=1)
        names = _text_vector(payload, "names_" + block)
        if len(names) != ncomp:
            raise ValueError(
                "historical Uniform block %r variable index is inconsistent" % block)
        if np.asarray(payload["state_" + block]).size != ncomp * nx * ny:
            raise ValueError(
                "historical Uniform block %r state has the wrong size" % block)
        block_rows.append((block, ncomp, names))
    if np.asarray(payload["phi"]).size != nx * ny:
        raise ValueError("historical Uniform potential has the wrong size")

    histories = []
    for name in _text_vector(payload, "history_names"):
        keys = {
            "history_depth_" + name,
            "history_ncomp_" + name,
            "history_init_" + name,
            "history_policy_" + name,
            "history_stored_slots_" + name,
            "history_slot_dt_" + name,
        }
        required.update(keys)
        if not keys <= set(payload.files):
            raise ValueError(
                "historical Uniform history %r has an incomplete v2 payload" % name)
        depth = _scalar_int(payload, "history_depth_" + name, minimum=1)
        ncomp = _scalar_int(payload, "history_ncomp_" + name, minimum=1)
        initialized_value = np.asarray(payload["history_init_" + name])
        if initialized_value.shape != () or initialized_value.dtype.kind != "b":
            raise TypeError(
                "historical Uniform history %r initialized flag must be bool" % name)
        initialized = bool(initialized_value.item())
        policy, stored = _historical_dense_policy(payload, name, depth)
        if np.asarray(payload["history_slot_dt_" + name]).shape != (depth,):
            raise ValueError(
                "historical Uniform history %r dt index is truncated" % name)
        for slot in stored:
            key = "history_%s_%d" % (name, slot)
            required.add(key)
            if key not in payload.files \
                    or np.asarray(payload[key]).size != ncomp * nx * ny:
                raise ValueError(
                    "historical Uniform history %r slot %d is missing or malformed"
                    % (name, slot))
        if initialized != (macro_step > 0):
            raise ValueError(
                "historical Uniform history %r initialization cannot be derived "
                "from its accepted clock" % name)
        histories.append((name, depth, policy, stored, initialized))

    cache_nodes = _int_vector(payload, "cache_nodes")
    cache_names = _text_vector(payload, "cache_names")
    if len(cache_nodes) != len(set(cache_nodes)) or len(cache_names) != len(cache_nodes):
        raise ValueError("historical Uniform cache index is inconsistent")
    for node in cache_nodes:
        if node < 0:
            raise ValueError("historical Uniform cache node ids must be non-negative")
        keys = {
            "cache_ncomp_%d" % node,
            "cache_ngrow_%d" % node,
            "cache_last_update_%d" % node,
            "cache_accum_dt_%d" % node,
            "cache_value_%d" % node,
        }
        required.update(keys)
        if not keys <= set(payload.files):
            raise ValueError(
                "historical Uniform cache node %d has an incomplete v2 payload" % node)
        ncomp = _scalar_int(payload, "cache_ncomp_%d" % node, minimum=1)
        _scalar_int(payload, "cache_ngrow_%d" % node)
        _scalar_int(payload, "cache_last_update_%d" % node, minimum=-1)
        _scalar_float(payload, "cache_accum_dt_%d" % node)
        if np.asarray(payload["cache_value_%d" % node]).size != ncomp * nx * ny:
            raise ValueError(
                "historical Uniform cache node %d has the wrong value size" % node)

    expected = required | {MANIFEST_KEY, IDENTITY_KEY}
    if set(payload.files) != expected:
        raise ValueError(
            "historical Uniform v2 checkpoint keys differ from its exact supported schema")
    return {
        "time": time,
        "macro_step": macro_step,
        "nx": nx,
        "ny": ny,
        "program_hash": program_hash,
        "blocks": tuple(block_rows),
        "histories": tuple(histories),
    }


__all__: list[str] = []
