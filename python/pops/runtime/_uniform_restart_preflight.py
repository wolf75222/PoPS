"""Semantic completeness checks for the strict Uniform checkpoint payload."""
from __future__ import annotations

from typing import Any


def preflight_uniform_restart(payload: Any) -> None:
    """Validate every dynamic history/cache key before restart mutates native state."""
    files = set(getattr(payload, "files", ()))
    required = {
        "program_hash", "history_names", "cache_nodes", "cache_names",
        "temporal_restart_state",
    }
    missing = sorted(required - files)
    if missing:
        raise ValueError("restart : strict Uniform checkpoint is missing %s" % ", ".join(missing))

    program_hash = str(payload["program_hash"])
    if len(program_hash) != 64:
        raise ValueError("restart : compiled Program hash must contain exactly 64 hexadecimal digits")
    try:
        bytes.fromhex(program_hash)
    except ValueError:
        raise ValueError("restart : compiled Program hash is not hexadecimal") from None

    history_names = [str(name) for name in payload["history_names"]]
    if len(history_names) != len(set(history_names)):
        raise ValueError("restart : strict Uniform history index contains duplicate names")
    for name in history_names:
        keys = {
            "history_depth_" + name, "history_ncomp_" + name,
            "history_init_" + name, "history_policy_" + name,
            "history_stored_slots_" + name, "history_slot_dt_" + name,
        }
        absent = sorted(keys - files)
        if absent:
            raise ValueError(
                "restart : history '%s' has an incomplete strict manifest (%s)"
                % (name, ", ".join(absent)))
        depth = int(payload["history_depth_" + name])
        ncomp = int(payload["history_ncomp_" + name])
        if depth <= 0 or ncomp <= 0:
            raise ValueError("restart : history '%s' depth/ncomp must be positive" % name)
        slots = [int(slot) for slot in payload["history_stored_slots_" + name]]
        if len(slots) != len(set(slots)) or any(slot < 0 or slot >= depth for slot in slots):
            raise ValueError("restart : history '%s' stored-slot index is invalid" % name)
        for slot in slots:
            key = "history_%s_%d" % (name, slot)
            if key not in files:
                raise ValueError("restart : history '%s' is missing stored slot %d" % (name, slot))
        if len(payload["history_slot_dt_" + name]) != depth:
            raise ValueError("restart : history '%s' dt index is truncated" % name)

    cache_nodes = [int(node) for node in payload["cache_nodes"]]
    cache_names = [str(name) for name in payload["cache_names"]]
    if len(cache_nodes) != len(set(cache_nodes)) or len(cache_names) != len(cache_nodes):
        raise ValueError("restart : strict Uniform cache index is inconsistent")
    for node in cache_nodes:
        keys = {
            "cache_ncomp_%d" % node, "cache_ngrow_%d" % node,
            "cache_last_update_%d" % node, "cache_accum_dt_%d" % node,
            "cache_value_%d" % node,
        }
        absent = sorted(keys - files)
        if absent:
            raise ValueError(
                "restart : scheduled cache node %d has an incomplete strict manifest (%s)"
                % (node, ", ".join(absent)))


__all__ = ["preflight_uniform_restart"]
