"""AM-06 level-count contract: campaign support is three materialized levels.

Does not import pops or read a PoPS output. Does not require a live runtime.
"""
from __future__ import annotations

SUPPORTED_LEVELS = 3


def supported_level_count() -> int:
    """Documented maximum materialized AMR level count for this campaign."""
    return int(SUPPORTED_LEVELS)
