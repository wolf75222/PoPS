"""Time-policy bricks and stable physical component roles.

Local IMEX/source-implicit policies remain descriptors because they select a per-block spatial/time
backend. Global solves are explicit :class:`pops.Program` graphs; reusable integration programs live
under :mod:`pops.lib.time` without introducing physics-specific time presets.
"""
from __future__ import annotations

from pops.runtime._bricks_time_imex import (  # noqa: F401
    IMEX,
    IMEXRK,
    SourceImplicit,
    SourceImplicitBE,
    _norm_implicit,
)


__all__ = [
    "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE", "_norm_implicit",
]
