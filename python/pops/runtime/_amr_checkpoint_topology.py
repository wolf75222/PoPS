"""Topology payload helpers for strict AMR checkpoint/restart."""


def owner_ranks_for_boxes(payload, boxes, level_count):
    """Return the owner rank aligned with each level-tagged fine patch box."""
    import numpy as np

    per_level = {
        level: list(np.asarray(payload["dmap_%d" % level], dtype=np.int64))
        for level in range(level_count) if ("dmap_%d" % level) in payload
    }
    cursor = {level: 0 for level in range(level_count)}
    owners = []
    for level, _ilo, _jlo, _ihi, _jhi in boxes:
        if level not in per_level:
            raise ValueError(
                "restart: checkpoint lacks owner-rank map for AMR level %d" % level)
        index = cursor[level]
        if index >= len(per_level[level]):
            raise ValueError(
                "restart: owner-rank map for AMR level %d is truncated" % level)
        owners.append(int(per_level[level][index]))
        cursor[level] = index + 1
    return owners


__all__ = ["owner_ranks_for_boxes"]
