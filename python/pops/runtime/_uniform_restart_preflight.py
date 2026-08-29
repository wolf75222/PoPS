"""Semantic completeness checks for the strict Uniform checkpoint payload."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

from pops._checkpoint_migration_protocol import validate_checkpoint_migration_provenance


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
    if value.shape != () or value.dtype.kind != "f" or value.dtype.itemsize != 8:
        raise TypeError("restart : %s must be an exact binary64 scalar" % key)
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("restart : %s must be finite" % key)
    if minimum is not None and result < minimum:
        raise ValueError("restart : %s must be >= %s" % (key, minimum))
    return result


def _float_array(payload: Any, key: str) -> Any:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim == 0 or value.dtype != np.dtype(np.float64) or not value.flags.c_contiguous:
        raise TypeError("restart : %s must be an exact C-contiguous binary64 array" % key)
    return value


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


def _bool_vector(payload: Any, key: str) -> tuple[bool, ...]:
    import numpy as np

    value = np.asarray(payload[key])
    if value.ndim != 1 or value.dtype.kind != "b":
        raise TypeError("restart : %s must be an exact boolean vector" % key)
    return tuple(bool(item) for item in value.tolist())


def preflight_uniform_restart(payload: Any) -> None:
    """Validate every dynamic history/cache key before restart mutates native state."""
    from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
    from pops.output._checkpoint_contract import (
        IDENTITY_KEY,
        MANIFEST_KEY,
        PROGRAM_PERSISTENT_CHECKPOINT_KEY,
        PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY,
        PROGRAM_PERSISTENT_PLAN_DIGEST_KEY,
        PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY,
        PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY,
        PROGRAM_PERSISTENT_SLOT_COUNT_KEY,
        program_persistent_checkpoint_from_payload,
    )
    from pops.runtime._program_cadence_checkpoint import (
        PROGRAM_CADENCE_CHECKPOINT_KEYS,
    )
    from pops.runtime._checkpoint_manifest import require_exact_payload_version
    from pops.runtime._system_io_history import (
        _history_level_key,
        _history_slot_key,
        validate_history_slot_dt_payload,
    )

    files = _files(payload)
    required = {
        "pops_checkpoint_version",
        "t",
        "macro_step",
        "pops_spatial_contract",
        "pops_embedded_boundary_contract",
        "program_hash",
        "history_names",
        "cache_slots",
        "cache_plan_schema",
        "cache_plan_digest",
        "cache_names",
        "cache_valid",
        "cache_cold",
        "temporal_restart_state",
    } | PROGRAM_CADENCE_CHECKPOINT_KEYS
    missing = sorted(required - files)
    if missing:
        raise ValueError("restart : strict Uniform checkpoint is missing %s" % ", ".join(missing))

    require_exact_payload_version(
        payload,
        key="pops_checkpoint_version",
        expected=UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        runtime_kind="Uniform",
    )
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

    # Absence is the normal native checkpoint case.  Presence is an offline Uniform-v2 migration
    # record and must be fully consumed before the archive can approach native restore.
    validate_checkpoint_migration_provenance(payload)

    allowed = {
        "pops_checkpoint_version",
        "t",
        "macro_step",
        "abi_key",
        "blocks",
        "temporal_restart_state",
        "pops_spatial_contract",
        "pops_embedded_boundary_contract",
        "program_hash",
        "field_provider_slots",
        "phi",
        "auxiliary_checkpoint",
        "checkpoint_migration",
        "history_names",
        "cache_slots",
        "cache_plan_schema",
        "cache_plan_digest",
        "cache_names",
        "cache_valid",
        "cache_cold",
        "runtime_consumer_graph",
        "runtime_consumer_cursors",
        "runtime_consumer_diagnostics",
        MANIFEST_KEY,
        IDENTITY_KEY,
        PROGRAM_PERSISTENT_CHECKPOINT_KEY,
        PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY,
        PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY,
        PROGRAM_PERSISTENT_PLAN_DIGEST_KEY,
        PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY,
        PROGRAM_PERSISTENT_SLOT_COUNT_KEY,
        *PROGRAM_CADENCE_CHECKPOINT_KEYS,
    }
    if "blocks" in files:
        blocks = _text_vector(payload, "blocks", unique=True)
        for block in blocks:
            keys = {"ncomp_" + block, "names_" + block, "state_" + block}
            absent = sorted(keys - files)
            if absent:
                raise ValueError(
                    "restart : block '%s' has an incomplete strict manifest (%s)"
                    % (block, ", ".join(absent))
                )
            allowed.update(keys)
    if "field_provider_slots" in files:
        field_slots = _text_vector(payload, "field_provider_slots", unique=True)
        for index in range(len(field_slots)):
            key = "field_potential_%d" % index
            if key not in files:
                raise ValueError(
                    "restart : field provider %d is missing its strict potential" % index
                )
            allowed.add(key)

    history_names = _text_vector(payload, "history_names", unique=True)
    for name in history_names:
        keys = {
            "history_depth_" + name,
            "history_ncomp_" + name,
            "history_policy_" + name,
            "history_requested_stored_slots_" + name,
            "history_stored_slots_" + name,
            "history_storage_mode_" + name,
        }
        absent = sorted(keys - files)
        if absent:
            raise ValueError(
                "restart : history '%s' has an incomplete strict manifest (%s)"
                % (name, ", ".join(absent))
            )
        allowed.update(keys)
        depth = _integer_scalar(payload, "history_depth_" + name, minimum=1)
        _integer_scalar(payload, "history_ncomp_" + name, minimum=1)
        _text_scalar(payload, "history_policy_" + name)
        _text_scalar(payload, "history_storage_mode_" + name)
        requested = _integer_vector(payload, "history_requested_stored_slots_" + name)
        stored = _integer_vector(payload, "history_stored_slots_" + name)
        for label, slots in (("requested", requested), ("stored", stored)):
            if slots != tuple(sorted(set(slots))) or any(
                slot < 0 or slot >= depth for slot in slots
            ):
                raise ValueError("restart : history '%s' %s-slot index is invalid" % (name, label))
        regrid_key = "history_regrid_steps_" + name
        if regrid_key in files:
            allowed.add(regrid_key)
            regrid_steps = _integer_vector(payload, regrid_key)
            if regrid_steps != tuple(sorted(set(regrid_steps))) or any(
                step < 0 for step in regrid_steps
            ):
                raise ValueError("restart : history '%s' replay-step index is invalid" % name)

        levels_key = "history_levels_" + name
        if levels_key in files:
            allowed.add(levels_key)
            levels = _integer_vector(payload, levels_key)
            if not levels or levels != tuple(sorted(set(levels))) or any(level < 0 for level in levels):
                raise ValueError("restart : history '%s' hierarchy levels are invalid" % name)
        else:
            levels = (None,)
        for level in levels:
            metadata = {
                _history_level_key("history_init_", name, level),
                _history_level_key("history_fill_count_", name, level),
                _history_level_key("history_slot_dt_", name, level),
            }
            absent = sorted(metadata - files)
            if absent:
                raise ValueError(
                    "restart : history '%s' has incomplete level metadata (%s)"
                    % (name, ", ".join(absent))
                )
            allowed.update(metadata)
            initialized = _bool_scalar(payload, _history_level_key("history_init_", name, level))
            fill_count = _integer_scalar(
                payload, _history_level_key("history_fill_count_", name, level), minimum=0
            )
            if fill_count > depth or initialized != (fill_count > 0):
                raise ValueError(
                    "restart : history '%s' has inconsistent initialized/fill-count metadata" % name
                )
            validate_history_slot_dt_payload(payload, name, depth, fill_count, level=level)
            for slot in stored:
                key = _history_slot_key(name, level, slot)
                if key not in files:
                    raise ValueError(
                        "restart : history '%s' is missing stored slot %d" % (name, slot)
                    )
                allowed.add(key)

    cache_slots = _integer_vector(payload, "cache_slots")
    cache_plan_schema = _text_scalar(payload, "cache_plan_schema")
    cache_plan_digest = _text_scalar(payload, "cache_plan_digest")
    cache_names = _text_vector(payload, "cache_names", unique=True)
    cache_valid = _bool_vector(payload, "cache_valid")
    cache_cold = _bool_vector(payload, "cache_cold")
    if cache_plan_schema != "program-resource-plan:v1":
        raise ValueError("restart : strict Uniform cache plan schema is unsupported")
    try:
        digest_bytes = bytes.fromhex(cache_plan_digest)
    except ValueError:
        digest_bytes = b""
    if len(cache_plan_digest) != 64 or digest_bytes.hex() != cache_plan_digest:
        raise ValueError("restart : strict Uniform cache plan digest is not canonical SHA-256")
    if (
        cache_slots != tuple(range(len(cache_slots)))
        or len(cache_names) != len(cache_slots)
        or len(cache_valid) != len(cache_slots)
        or len(cache_cold) != len(cache_slots)
    ):
        raise ValueError("restart : strict Uniform cache index is inconsistent")
    if any(valid == cold for valid, cold in zip(cache_valid, cache_cold, strict=True)):
        raise ValueError("restart : strict Uniform cache validity/cold state is inconsistent")
    if any(cache_valid) and macro_step == 0:
        raise ValueError("restart : a step-zero checkpoint cannot contain a valid scheduled cache")
    for slot, cache_name, valid, cold in zip(
        cache_slots, cache_names, cache_valid, cache_cold, strict=True
    ):
        if not cache_name:
            raise ValueError("restart : scheduled cache slot %d has no static identity" % slot)
        keys = {
            "cache_ncomp_%d" % slot,
            "cache_ngrow_%d" % slot,
            "cache_last_update_%d" % slot,
            "cache_accum_dt_%d" % slot,
        }
        if valid:
            keys.add("cache_value_%d" % slot)
        absent = sorted(keys - files)
        if absent:
            raise ValueError(
                "restart : scheduled cache slot %d has an incomplete strict manifest (%s)"
                % (slot, ", ".join(absent))
            )
        allowed.update(keys)
        ncomp = _integer_scalar(payload, "cache_ncomp_%d" % slot, minimum=0)
        ngrow = _integer_scalar(payload, "cache_ngrow_%d" % slot, minimum=0)
        last_update = _integer_scalar(payload, "cache_last_update_%d" % slot)
        _float_scalar(payload, "cache_accum_dt_%d" % slot, minimum=0.0)
        if valid:
            value = _float_array(payload, "cache_value_%d" % slot)
            if value.size == 0:
                raise ValueError(
                    "restart : scheduled cache slot %d has an empty accepted value" % slot
                )
            if ncomp < 1 or last_update < 0 or last_update >= macro_step:
                raise ValueError(
                    "restart : scheduled cache slot %d has invalid accepted metadata" % slot
                )
        elif not cold or ncomp != 0 or ngrow != 0 or last_update != -1:
            raise ValueError(
                "restart : cold scheduled cache slot %d has fabricated value metadata" % slot
            )

    persistent_keys = {
        PROGRAM_PERSISTENT_CHECKPOINT_KEY,
        PROGRAM_PERSISTENT_CHECKPOINT_SCHEMA_KEY,
        PROGRAM_PERSISTENT_PLAN_SCHEMA_KEY,
        PROGRAM_PERSISTENT_PLAN_DIGEST_KEY,
        PROGRAM_PERSISTENT_PLAN_MAXIMUM_BYTES_KEY,
        PROGRAM_PERSISTENT_SLOT_COUNT_KEY,
    }
    persistent_present = persistent_keys.intersection(files)
    if persistent_present:
        if persistent_present != persistent_keys:
            missing = sorted(persistent_keys - persistent_present)
            raise ValueError(
                "restart : Program persistent checkpoint envelope is incomplete (%s)"
                % ", ".join(missing)
            )
        # Decode and authenticate POPSPVS1 before the native restart transaction.  This is kept in
        # the preflight even though the sealed manifest also inspects it, because callers of this
        # semantic guard may provide an in-memory payload without a manifest.
        program_persistent_checkpoint_from_payload(payload)

    unknown = sorted(files - allowed)
    if unknown:
        raise ValueError(
            "restart : strict Uniform checkpoint has unknown archive members %s"
            % ", ".join(unknown)
        )


__all__ = ["preflight_uniform_restart"]
