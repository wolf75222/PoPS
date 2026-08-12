"""Topology payload helpers for strict AMR checkpoint/restart."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordedRankTopology:
    """Canonical Program image and its authenticated recorded ownership metadata."""

    program_state: bytes
    source_rank_count: int
    level_owner_ranks: tuple[tuple[int, ...], ...]


def recorded_rank_topology(payload: Any, level_count: int, rank_count: int) -> RecordedRankTopology:
    """Authenticate the rank-independent Program image and one canonical owner map per level."""
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

    level_maps = []
    for level in range(level_count):
        dmap_key = "dmap_%d" % level
        if dmap_key not in payload:
            raise ValueError("restart: AMR checkpoint lacks owner map for level %d" % level)
        owner_map = np.asarray(payload[dmap_key])
        if owner_map.dtype.kind not in "iu" or owner_map.ndim != 1:
            raise TypeError("restart: AMR owner map for level %d must be an integer vector" % level)
        owners = tuple(int(owner) for owner in owner_map)
        if any(owner < 0 or owner >= rank_count for owner in owners):
            raise ValueError(
                "restart: AMR owner map for level %d contains an owner outside [0, %d)"
                % (level, rank_count)
            )
        level_maps.append(owners)
    return RecordedRankTopology(state.tobytes(), rank_count, tuple(level_maps))


def owner_ranks_for_boxes(payload, boxes, level_count):
    """Return the owner rank aligned with each level-tagged fine patch box."""
    import numpy as np

    per_level = {
        level: list(np.asarray(payload["dmap_%d" % level], dtype=np.int64))
        for level in range(level_count)
        if ("dmap_%d" % level) in payload
    }
    cursor = {level: 0 for level in range(level_count)}
    owners = []
    for box in boxes:
        level = box[0]
        if level not in per_level:
            raise ValueError("restart: checkpoint lacks owner-rank map for AMR level %d" % level)
        index = cursor[level]
        if index >= len(per_level[level]):
            raise ValueError("restart: owner-rank map for AMR level %d is truncated" % level)
        owners.append(int(per_level[level][index]))
        cursor[level] = index + 1
    return owners


__all__ = ["RecordedRankTopology", "owner_ranks_for_boxes", "recorded_rank_topology"]
