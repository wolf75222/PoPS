"""Exact AMR distribution topology persisted by checkpoint payload v12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_MODES = frozenset({"replicated", "partitioned"})


@dataclass(frozen=True, slots=True)
class RecordedRankTopology:
    program_state: bytes
    source_rank_count: int
    level_distribution_modes: tuple[str, ...]
    level_owner_ranks: tuple[tuple[int, ...], ...]


def _exact_mode(payload: Any, level: int) -> str:
    import numpy as np

    key = "distribution_mode_%d" % level
    if key not in payload:
        raise ValueError("restart: AMR checkpoint lacks distribution mode for level %d" % level)
    raw = np.asarray(payload[key])
    if raw.ndim != 0 or raw.dtype.kind not in "US":
        raise TypeError("restart: AMR distribution mode for level %d must be a text scalar" % level)
    mode = str(raw.item())
    if mode not in _MODES:
        raise ValueError("restart: AMR distribution mode for level %d is unknown" % level)
    return mode


def recorded_rank_topology(payload: Any, level_count: int, rank_count: int) -> RecordedRankTopology:
    """Authenticate every active mode and its canonical owner map without inference."""
    import numpy as np

    if isinstance(level_count, bool) or not isinstance(level_count, int) or level_count < 1:
        raise ValueError("restart: AMR recorded level count must be a positive integer")
    if isinstance(rank_count, bool) or not isinstance(rank_count, int) or rank_count < 1:
        raise ValueError("restart: AMR recorded rank count must be a positive integer")

    state_key = "program_accepted_state"
    if state_key not in payload:
        raise ValueError("restart: AMR checkpoint lacks its canonical accepted Program state")
    state = np.asarray(payload[state_key])
    if state.dtype != np.dtype("uint8") or state.ndim != 1:
        raise TypeError("restart: AMR accepted Program state must be a uint8 vector")

    expected = {
        *("distribution_mode_%d" % level for level in range(level_count)),
        *("dmap_%d" % level for level in range(level_count)),
    }
    files = payload.files if hasattr(payload, "files") else payload.keys()
    unexpected = sorted(
        key
        for key in files
        if (key.startswith("distribution_mode_") or key.startswith("dmap_")) and key not in expected
    )
    if unexpected:
        raise ValueError("restart: AMR checkpoint has surplus distribution topology member(s) %r" % unexpected)

    modes = []
    level_maps = []
    for level in range(level_count):
        mode = _exact_mode(payload, level)
        dmap_key = "dmap_%d" % level
        if dmap_key not in payload:
            raise ValueError("restart: AMR checkpoint lacks owner map for level %d" % level)
        owner_map = np.asarray(payload[dmap_key])
        if owner_map.dtype != np.dtype("int64") or owner_map.ndim != 1:
            raise TypeError("restart: AMR owner map for level %d must be an int64 vector" % level)
        owners = tuple(int(owner) for owner in owner_map)
        if mode == "replicated":
            if owners:
                raise ValueError("restart: replicated AMR level %d must have an empty owner map" % level)
        elif not owners:
            raise ValueError("restart: partitioned AMR level %d must have an owner map" % level)
        if any(owner < 0 or owner >= rank_count for owner in owners):
            raise ValueError(
                "restart: AMR owner map for level %d contains an owner outside [0, %d)"
                % (level, rank_count)
            )
        modes.append(mode)
        level_maps.append(owners)
    return RecordedRankTopology(state.tobytes(), rank_count, tuple(modes), tuple(level_maps))


def owner_ranks_for_boxes(topology: RecordedRankTopology, boxes, level_count: int) -> tuple[int, ...]:
    """Return native rebuild witnesses: ``-1`` only for authenticated replicated fine levels."""
    if not isinstance(topology, RecordedRankTopology):
        raise TypeError("restart: AMR owner alignment requires RecordedRankTopology")
    if level_count != len(topology.level_owner_ranks) or level_count != len(topology.level_distribution_modes):
        raise ValueError("restart: AMR recorded distribution topology has the wrong active depth")
    cursor = {level: 0 for level in range(level_count)}
    owners = []
    for box in boxes:
        level = box[0]
        if type(level) is not int or level < 1 or level >= level_count:
            raise ValueError("restart: AMR patch has an invalid fine level for owner alignment")
        mode = topology.level_distribution_modes[level]
        if mode == "replicated":
            owners.append(-1)
            continue
        index = cursor[level]
        ranks = topology.level_owner_ranks[level]
        if index >= len(ranks):
            raise ValueError("restart: owner-rank map for AMR level %d is truncated" % level)
        owners.append(ranks[index])
        cursor[level] = index + 1
    for level in range(1, level_count):
        mode = topology.level_distribution_modes[level]
        consumed = cursor[level]
        size = len(topology.level_owner_ranks[level])
        if mode == "partitioned" and consumed != size:
            raise ValueError("restart: owner-rank map for AMR level %d has surplus entries" % level)
        if mode == "replicated" and topology.level_owner_ranks[level]:
            raise ValueError("restart: replicated AMR level %d must have an empty owner map" % level)
    return tuple(owners)


__all__ = ["RecordedRankTopology", "owner_ranks_for_boxes", "recorded_rank_topology"]
