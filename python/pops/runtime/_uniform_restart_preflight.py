"""Semantic completeness checks for the strict Uniform checkpoint payload."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any


def _files(payload: Any) -> set[str]:
    stored = getattr(payload, "files", None)
    if stored is None:
        keys = getattr(payload, "keys", None)
        if not callable(keys):
            raise TypeError("restart : Uniform checkpoint exposes neither files nor keys()")
        stored = keys()
    elif callable(stored):
        stored = stored()
    if isinstance(stored, (str, bytes)) or not isinstance(stored, Iterable):
        raise TypeError(
            "restart : Uniform checkpoint files/keys() must return an iterable of names"
        )
    return {str(name) for name in stored}


def _integer_scalar(payload: Any, key: str, *, minimum: int | None = None) -> int:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "iu":
        raise TypeError("restart : %s must be an exact integer scalar" % key)
    result = int(value.item())
    if minimum is not None and result < minimum:
        raise ValueError("restart : %s must be >= %d" % (key, minimum))
    return result


def _float_scalar(payload: Any, key: str, *, minimum: float | None = None) -> float:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind != "f":
        raise TypeError("restart : %s must be an exact floating scalar" % key)
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("restart : %s must be finite" % key)
    if minimum is not None and result < minimum:
        raise ValueError("restart : %s must be >= %s" % (key, minimum))
    return result


def _bool_scalar(payload: Any, key: str) -> bool:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind != "b":
        raise TypeError("restart : %s must be an exact boolean scalar" % key)
    return bool(value.item())


def _text_scalar(payload: Any, key: str) -> str:
    import numpy as np

    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind not in "US":
        raise TypeError("restart : %s must be an exact text scalar" % key)
    result = str(value.item())
    if not result:
        raise ValueError("restart : %s must be non-empty" % key)
    return result


def _text_vector(payload: Any, key: str, *, unique: bool) -> tuple[str, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise TypeError("restart : %s must be an exact text vector" % key)
    result = tuple(str(item) for item in value.tolist())
    if any(not item for item in result):
        raise ValueError("restart : %s must contain non-empty text" % key)
    if unique and len(result) != len(set(result)):
        raise ValueError("restart : %s must contain unique text" % key)
    return result


def _integer_vector(payload: Any, key: str) -> tuple[int, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise TypeError("restart : %s must be an exact integer vector" % key)
    return tuple(int(item) for item in value.tolist())


def preflight_uniform_restart(payload: Any) -> None:
    """Validate every dynamic history/cache key before restart mutates native state."""
    from pops.runtime._program_cadence_checkpoint import (
        PROGRAM_CADENCE_CHECKPOINT_KEYS,
    )
    from pops.runtime._system_io_history import validate_history_slot_dt_payload

    files = _files(payload)
    required = {
        "t",
        "macro_step",
        "pops_spatial_contract",
        "pops_embedded_boundary_contract",
        "program_hash",
        "history_names",
        "cache_nodes",
        "cache_names",
        "temporal_restart_state",
    } | PROGRAM_CADENCE_CHECKPOINT_KEYS
    missing = sorted(required - files)
    if missing:
        raise ValueError("restart : strict Uniform checkpoint is missing %s" % ", ".join(missing))

    _float_scalar(payload, "t")
    macro_step = _integer_scalar(payload, "macro_step", minimum=0)
    program_hash = _text_scalar(payload, "program_hash")
    if len(program_hash) != 64:
        raise ValueError(
            "restart : compiled Program hash must contain exactly 64 hexadecimal digits"
        )
    try:
        bytes.fromhex(program_hash)
    except ValueError:
        raise ValueError("restart : compiled Program hash is not hexadecimal") from None

    history_names = _text_vector(payload, "history_names", unique=True)
    for name in history_names:
        keys = {
            "history_depth_" + name,
            "history_ncomp_" + name,
            "history_init_" + name,
            "history_fill_count_" + name,
            "history_policy_" + name,
            "history_requested_stored_slots_" + name,
            "history_stored_slots_" + name,
            "history_storage_mode_" + name,
            "history_slot_dt_" + name,
        }
        absent = sorted(keys - files)
        if absent:
            raise ValueError(
                "restart : history '%s' has an incomplete strict manifest (%s)"
                % (name, ", ".join(absent))
            )
        depth = _integer_scalar(payload, "history_depth_" + name, minimum=1)
        _integer_scalar(payload, "history_ncomp_" + name, minimum=1)
        initialized = _bool_scalar(payload, "history_init_" + name)
        fill_count = _integer_scalar(payload, "history_fill_count_" + name, minimum=0)
        if fill_count > depth or initialized != (fill_count > 0):
            raise ValueError(
                "restart : history '%s' has inconsistent initialized/fill-count metadata" % name
            )
        _text_scalar(payload, "history_policy_" + name)
        _text_scalar(payload, "history_storage_mode_" + name)
        requested = _integer_vector(payload, "history_requested_stored_slots_" + name)
        stored = _integer_vector(payload, "history_stored_slots_" + name)
        for label, slots in (("requested", requested), ("stored", stored)):
            if slots != tuple(sorted(set(slots))) or any(
                slot < 0 or slot >= depth for slot in slots
            ):
                raise ValueError("restart : history '%s' %s-slot index is invalid" % (name, label))
        for slot in stored:
            key = "history_%s_%d" % (name, slot)
            if key not in files:
                raise ValueError("restart : history '%s' is missing stored slot %d" % (name, slot))
        validate_history_slot_dt_payload(payload, name, depth, fill_count)
        regrid_key = "history_regrid_steps_" + name
        if regrid_key in files:
            regrid_steps = _integer_vector(payload, regrid_key)
            if regrid_steps != tuple(sorted(set(regrid_steps))) or any(
                step < 0 for step in regrid_steps
            ):
                raise ValueError("restart : history '%s' replay-step index is invalid" % name)

    cache_nodes = _integer_vector(payload, "cache_nodes")
    cache_names = _text_vector(payload, "cache_names", unique=False)
    if (
        cache_nodes != tuple(sorted(set(cache_nodes)))
        or any(node < 0 for node in cache_nodes)
        or len(cache_names) != len(cache_nodes)
    ):
        raise ValueError("restart : strict Uniform cache index is inconsistent")
    if cache_nodes and macro_step == 0:
        raise ValueError("restart : a step-zero checkpoint cannot contain a valid scheduled cache")
    for node in cache_nodes:
        keys = {
            "cache_ncomp_%d" % node,
            "cache_ngrow_%d" % node,
            "cache_last_update_%d" % node,
            "cache_accum_dt_%d" % node,
            "cache_value_%d" % node,
        }
        absent = sorted(keys - files)
        if absent:
            raise ValueError(
                "restart : scheduled cache node %d has an incomplete strict manifest (%s)"
                % (node, ", ".join(absent))
            )
        _integer_scalar(payload, "cache_ncomp_%d" % node, minimum=1)
        _integer_scalar(payload, "cache_ngrow_%d" % node, minimum=0)
        last_update = _integer_scalar(payload, "cache_last_update_%d" % node, minimum=0)
        if last_update >= macro_step:
            raise ValueError(
                "restart : scheduled cache node %d last update is not an accepted prior step" % node
            )
        _float_scalar(payload, "cache_accum_dt_%d" % node, minimum=0.0)


__all__ = ["preflight_uniform_restart"]
